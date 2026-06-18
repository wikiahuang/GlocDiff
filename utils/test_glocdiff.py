"""
Closed-loop GlocDiff rollout in iGibson.

The robot has no physical body: the "robot" is the iGibson viewer/render camera, which
is teleported step by step (matches the camera-only visual-navigation setup used to 
collect training data). 

TODO before running:
  - Fill in config/test_glocdiff.yaml: checkpoint_path, testdataset,
    trav_maps_path, scene_path, test_scenes, traj_index_range.
"""
import logging
import os
import sys
import time

import numpy as np
import cv2
import networkx as nx
import yaml
import matplotlib.pyplot as plt
from PIL import Image

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "diffusion_policy"))

from model.glocdiff import glocdiff
from model.deepfloor_net import deep_floor_net
from model.depth_latent_processor import depth_latent_processor
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

import data_utils

from igibson.simulator import Simulator
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings

import diffusers as diffusers_pkg

ACTION_STATS = {"min": np.array([-2.5, -4]), "max": np.array([5, 4])}

# Of the len_traj_pred predicted waypoints, only execute this many before replanning
# (leaves the tail of the prediction as unused lookahead, matching the old script).
EXECUTE_HORIZON = 20

# Navigation-relevant logger (position/distance/collision/arrival). Console only shows
# this logger's messages; everything else (iGibson, torch, diffusers, ...) is routed to
# the log file only, see setup_logging().
logger = logging.getLogger("glocdiff_test")


def setup_logging(state_save_dir):
    """Route this script's nav-relevant messages to console+file, and silence every other
    library's logging on the console (still recorded to file for debugging)."""
    os.makedirs(state_save_dir, exist_ok=True)
    log_path = os.path.join(state_save_dir, "test_glocdiff.log")
    file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = [file_handler]

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.setLevel(logging.INFO)
    logger.handlers = [console_handler, file_handler]
    logger.propagate = False  # don't also duplicate into root's file handler
    return log_path


# --------------------------------------------------------------------------
# diffusion inference (single-process variant of train_utils.model_output,
# which hardcodes os.environ['LOCAL_RANK'] for DDP training)
# --------------------------------------------------------------------------

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    return ndata * (stats["max"] - stats["min"]) + stats["min"]


def get_action(diffusion_output, action_stats=ACTION_STATS):
    ndeltas = diffusion_output.reshape(diffusion_output.shape[0], -1, 2)
    ndeltas = ndeltas.detach().cpu().numpy()
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return torch.from_numpy(actions).float().to(diffusion_output.device)


def model_output_single(model, noise_scheduler, depth_cond, local_cond, pred_horizon, action_dim, device):
    diffusion_output = torch.randn((depth_cond.shape[0], pred_horizon, action_dim), device=device)
    for k in noise_scheduler.timesteps:
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=depth_cond,
            local_cond=local_cond,
        )
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred, timestep=int(k), sample=diffusion_output
        ).prev_sample
    return get_action(diffusion_output)


def predict_waypoints(model, depth_latent, floorplan_tensor, cur_pos, cur_heading, goal_pos,
                       shortest_actions, noise_scheduler, len_traj_pred, device):
    """One inference pass: depth latent -> fused condition -> diffusion-denoised local waypoints."""
    with torch.no_grad():
        depth_cond = model("depth_latent_processor", depth_latent=depth_latent)
        depth_cond = model(
            "deep_floor_net",
            floor_plan=floorplan_tensor,
            cur_pos=cur_pos,
            cur_heading=cur_heading,
            goal_pos=goal_pos,
            depth_cond=depth_cond,
        )
        actions = model_output_single(
            model, noise_scheduler, depth_cond, shortest_actions, len_traj_pred, 2, device
        )
    return actions


def load_model(checkpoint_path, config, device):
    deepfloor_net = deep_floor_net(
        floorplan_encoder=config["floorplan_encoder"],
        floorplan_encoding_size=config["encoding_size"],
    )
    depth_processor = depth_latent_processor(
        depth_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        local_cond_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    model = glocdiff(
        depth_latent_processor=depth_processor,
        noise_pred_net=noise_pred_net,
        deep_floor_net=deepfloor_net,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    return model, noise_scheduler


# --------------------------------------------------------------------------
# online shortest-path local condition (A* over a graph built from the
# floor plan image; GlocDiffDataset reads this precomputed from disk instead)
# --------------------------------------------------------------------------

def l2_distance(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))


def build_graph(floorplan_path):
    floorplan = np.array(Image.open(floorplan_path).convert("L"))
    floorplan_size = int(floorplan.shape[0] / 10)
    floorplan = cv2.resize(floorplan, (floorplan_size, floorplan_size))
    floorplan = cv2.erode(floorplan, np.ones((3, 3)))
    floorplan[floorplan < 127] = 0
    floorplan[floorplan >= 127] = 255

    g = nx.Graph()
    for i in range(floorplan_size):
        for j in range(floorplan_size):
            if floorplan[i, j] == 0:
                continue
            g.add_node((i, j))
            neighbors = [
                (i - 1, j - 1), (i, j - 1), (i + 1, j - 1),
                (i - 1, j), (i + 1, j),
                (i - 1, j + 1), (i, j + 1), (i + 1, j + 1),
            ]
            for n in neighbors:
                if 0 <= n[0] < floorplan_size and 0 <= n[1] < floorplan_size and floorplan[n[0], n[1]] > 0:
                    wgt = l2_distance(n, (i, j))
                    near_wall = any(
                        0 <= nn_[0] < floorplan_size and 0 <= nn_[1] < floorplan_size and floorplan[nn_[0], nn_[1]] == 0
                        for nn_ in [(n[0] - 1, n[1] - 1), (n[0], n[1] - 1), (n[0] + 1, n[1] - 1), (n[0] - 1, n[1])]
                    )
                    if near_wall:
                        wgt += 0.5
                    g.add_edge(n, (i, j), weight=wgt)

    largest_cc = max(nx.connected_components(g), key=len)
    return g.subgraph(largest_cc).copy(), floorplan_size


def shortest_path_waypoints(graph, floorplan_grid_size, cur_pos, goal_pos, len_traj_pred):
    """Real-time A* shortest path between cur_pos and goal_pos (meters), resampled to len_traj_pred (pos, heading) pairs."""
    start_pose = tuple((cur_pos * 10 + floorplan_grid_size / 2.0).astype(int))
    end_pose = tuple((goal_pos * 10 + floorplan_grid_size / 2.0).astype(int))
    start_pose, end_pose = (start_pose[1], start_pose[0]), (end_pose[1], end_pose[0])

    nodes = np.array(graph.nodes)
    if not graph.has_node(end_pose):
        closest = tuple(nodes[np.argmin(np.linalg.norm(nodes - end_pose, axis=1))])
        graph.add_edge(closest, end_pose, weight=l2_distance(closest, end_pose))
    if not graph.has_node(start_pose):
        closest = tuple(nodes[np.argmin(np.linalg.norm(nodes - start_pose, axis=1))])
        graph.add_edge(closest, start_pose, weight=l2_distance(closest, start_pose))

    try:
        path = np.array(nx.astar_path(graph, start_pose, end_pose, heuristic=l2_distance))
    except nx.NetworkXNoPath:
        return None
    path = (path - floorplan_grid_size / 2.0) * 0.1
    path = path[:, [1, 0]]

    sparse_path = path[::4]
    steps = 10
    dense_path = []
    for i in range(len(sparse_path) - 1):
        for j in range(steps):
            dense_path.append(sparse_path[i] * (steps - j) / steps + sparse_path[i + 1] * j / steps)
    dense_path = np.array(dense_path) if dense_path else np.repeat(sparse_path, steps, axis=0)

    headings = dense_path[6:]
    while len(headings) < len(dense_path):
        headings = np.concatenate((headings, [headings[-1]]), axis=0)
    traj = np.concatenate((dense_path, headings), axis=1)
    while traj.shape[0] < len_traj_pred:
        traj = np.concatenate((traj, [traj[-1]]), axis=0)
    return traj[:len_traj_pred]


def compute_local_actions(traj_data, len_traj_pred, metric_waypoint_spacing, waypoint_spacing):
    """Same transform as GlocDiffDataset._compute_actions: absolute (pos, heading) -> normalized local-frame waypoints."""
    positions = traj_data[:, :2].copy()
    yaw = traj_data[:, 2:].copy()
    waypoints = data_utils.to_local_coords(positions, positions[0], yaw[0])
    assert waypoints.shape == (len_traj_pred, 2), f"{waypoints.shape} != {(len_traj_pred, 2)}"
    waypoints = waypoints / (metric_waypoint_spacing * waypoint_spacing)
    return waypoints


# --------------------------------------------------------------------------
# camera-as-robot movement
# --------------------------------------------------------------------------

def check_collision(pos, travers_map):
    pos_in_map = (pos * 100 + np.array([travers_map.shape[0], travers_map.shape[1]]) // 2).astype(np.int16)
    if not (0 <= pos_in_map[1] < travers_map.shape[0] and 0 <= pos_in_map[0] < travers_map.shape[1]):
        return True
    return travers_map[pos_in_map[1], pos_in_map[0]] == 0


def sub_waypoints(subgoal, current_position, n=5):
    return [current_position + (subgoal - current_position) * (i + 1) / n for i in range(n)]


class FrameSaver:
    """Saves every rendered RGB frame for an episode to disk as 00000.png, 00001.png, ..."""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.count = 0

    def save(self, img):
        img.save(os.path.join(self.save_dir, f"{self.count:05d}.png"))
        self.count += 1


def camera_set_and_record(env, current_position, current_heading, image_size, frame_saver=None):
    # env.viewer is only created in gui_interactive/gui_non_interactive mode (None when
    # headless); it only drives the GUI window's own camera state and has no effect on
    # the actual offscreen render, which is controlled entirely by renderer.set_camera below.
    if env.viewer is not None:
        env.viewer.initial_pos = current_position
        env.viewer.initial_view_direction = current_heading - current_position
        env.viewer.reset_viewer()
    env.renderer.set_camera(current_position, current_heading, [0, 0, 1])
    frame = env.renderer.render(modes=("rgb",))
    img = Image.fromarray((255 * np.concatenate(frame, axis=1)[:, :, :3]).astype(np.uint8))
    if frame_saver is not None:
        frame_saver.save(img)
    resized_img = data_utils.resize_and_aspect_crop(img, image_size, data_utils.IMAGE_ASPECT_RATIO)
    env.step()
    return resized_img


def execute_initial_context(env, init_positions, init_headings, cur_obs_list, image_size, frame_saver=None):
    """Replay a few recorded frames to seed the observation context window."""
    for pos, heading in zip(init_positions, init_headings):
        resized_img = camera_set_and_record(env, pos, heading, image_size, frame_saver)
        cur_obs_list.append(resized_img.unsqueeze(0))
    return init_positions[-1], init_headings[-1]


def move_along_waypoints(env, current_position, trajectory, cur_obs_list, context_size, travers_map,
                          image_size, save_states, frame_saver=None):
    """Move the camera through a predicted local trajectory step by step, stopping on collision.
    Collision is checked at every fine-grained sub-waypoint (cheap position math, no rendering),
    but only the subgoal's final position (reached normally, or where collision stopped it) is
    actually rendered -- that's the only frame that ever entered cur_obs_list anyway, so
    rendering/saving the sub-waypoints in between would just be 5x the images for no benefit.
    A state row is appended to save_states for every rendered frame, matching the old script's
    granularity (one row per frame, not one per replanning step)."""
    collision = False
    current_heading_point = current_position
    for subgoal in trajectory:
        last_point = current_position
        for waypoint in sub_waypoints(subgoal, current_position):
            collision = check_collision(waypoint[:2], travers_map)
            current_position, current_heading_point = last_point, waypoint
            last_point = waypoint
            if collision:
                break
        resized_img = camera_set_and_record(env, current_position, current_heading_point, image_size, frame_saver)
        save_states.append(np.concatenate([current_position[:2], current_heading_point[:2], [collision]]))
        time.sleep(0.01)
        if len(cur_obs_list) >= context_size:
            cur_obs_list.pop(0)
        cur_obs_list.append(resized_img.unsqueeze(0))
        if collision:
            break
    return current_position, current_heading_point, collision


def recover_from_collision(env, current_position, current_heading_point, cur_obs_list, context_size,
                            travers_map, image_size, save_states, frame_saver=None, max_turns=8):
    """Rotate in place in 45-degree steps (random direction) until a collision-free heading is found."""
    current_yaw = np.arctan2(
        current_heading_point[1] - current_position[1], current_heading_point[0] - current_position[0]
    )
    direction = 1.0 if np.random.rand() < 0.5 else -1.0
    collision = True
    turns = 0
    while collision and turns < max_turns:
        turns += 1
        current_yaw += direction * (45 / 180 * np.pi)
        new_xy = current_position[:2] + np.array([np.cos(current_yaw), np.sin(current_yaw)]) * 0.02
        current_heading_point = np.append(new_xy, current_position[2])
        resized_img = camera_set_and_record(env, current_position, current_heading_point, image_size, frame_saver)
        save_states.append(np.concatenate([current_position[:2], current_heading_point[:2], [collision]]))
        for _ in range(context_size):
            if cur_obs_list:
                cur_obs_list.pop(0)
            cur_obs_list.append(resized_img.unsqueeze(0))
        time.sleep(0.01)
        collision = check_collision(current_heading_point[:2], travers_map)
    return current_heading_point, collision


def realign_to_shortest_path(env, current_position, current_heading_point, shortest_path,
                              cur_obs_list, image_size, frame_saver=None, angle_threshold_deg=150):
    """If the robot is facing away from the shortest path (e.g. nearly opposite), snap its
    heading to the path direction and re-render, instead of letting the model untangle it.
    Only the heading/observations are touched; shortest_actions stays path-relative and is
    unaffected (see compute_local_actions).

    Uses the chord from shortest_path's first point to its last (spanning the whole predicted
    horizon, ~1m+) rather than any single short segment: a 4-6cm or even ~24cm segment of the
    discretized A* path can be dominated by grid-quantization "staircase" noise near a corner,
    which made this fire on almost every step. The full-horizon chord still follows the
    obstacle-avoiding A* path (unlike a straight line to the goal, which can cut through walls)
    but averages out that local noise."""
    current_yaw = np.arctan2(
        current_heading_point[1] - current_position[1], current_heading_point[0] - current_position[0]
    )
    path_yaw = np.arctan2(
        shortest_path[-1][1] - shortest_path[0][1], shortest_path[-1][0] - shortest_path[0][0]
    )
    delta_yaw = (path_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi
    if abs(delta_yaw) * 180 / np.pi <= angle_threshold_deg:
        return current_heading_point

    new_xy = current_position[:2] + np.array([np.cos(path_yaw), np.sin(path_yaw)]) * 0.01
    current_heading_point = np.append(new_xy, current_position[2])
    resized_img = camera_set_and_record(env, current_position, current_heading_point, image_size, frame_saver)
    for _ in range(len(cur_obs_list)):
        cur_obs_list.pop(0)
        cur_obs_list.append(resized_img.unsqueeze(0))
    return current_heading_point


# --------------------------------------------------------------------------
# visualization
# --------------------------------------------------------------------------

def save_trajectory_plot(floorplan_path, save_states, goal_pos, save_path):
    """Overlay the episode's positions (red, collisions marked) and goal (blue star) on the
    scene's floor plan image. Positions are converted to pixels with the same meters-per-pixel
    convention check_collision() uses against the traversable map."""
    floorplan_img = np.array(Image.open(floorplan_path).convert("RGB"))
    h, w = floorplan_img.shape[:2]
    center = np.array([w, h]) / 2.0

    positions = save_states[:, :2]
    collided = save_states[:, 4].astype(bool)
    pos_px = positions * 100 + center
    goal_px = goal_pos * 100 + center

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(floorplan_img)
    ax.plot(pos_px[:, 0], pos_px[:, 1], "r-", linewidth=1, label="trajectory")
    ax.scatter(pos_px[~collided, 0], pos_px[~collided, 1], c="red", s=10)
    ax.scatter(pos_px[collided, 0], pos_px[collided, 1], c="orange", s=20, marker="x", label="collision")
    ax.scatter(pos_px[0, 0], pos_px[0, 1], c="green", s=80, marker="^", label="start")
    ax.scatter(goal_px[0], goal_px[1], c="blue", s=120, marker="*", label="goal")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------

CAMERA_EYE_HEIGHT = 0.85  # meters above the floor plane, matches the old script's convention


def get_floor_height(scene_path_root, scene_id, floor_idx):
    """Camera height for this floor: the floor's z-coordinate plus eye height. Without the
    offset the camera sits at/inside the floor mesh, clipping into it and rendering black
    in the lower half of the frame."""
    with open(os.path.join(scene_path_root, scene_id, "floors.txt")) as f:
        heights = sorted(float(line) for line in f if line.strip())
    return heights[floor_idx] + CAMERA_EYE_HEIGHT


def run_episode(env, model, noise_scheduler, depth_generator, graph, floorplan_grid_size,
                 floorplan_tensor, floorplan_path, travers_map, traj_data, height, config, device,
                 episode_img_dir, trajectory_plot_path):
    context_size = config["context_size"]
    len_traj_pred = config["len_traj_pred"]
    image_size = tuple(config["image_size"])
    metric_waypoint_spacing = config["metric_waypoint_spacing"]
    waypoint_spacing = config["waypoint_spacing"]
    arrive_th = config["arrive_th"]
    max_steps = config["max_steps"]

    init_positions = [np.array([traj_data[t][0], traj_data[t][1], height]) for t in range(context_size)]
    init_headings = [np.array([traj_data[t][2], traj_data[t][3], height]) for t in range(context_size)]
    goal_pos = np.array([traj_data[-1][0], traj_data[-1][1]])

    frame_saver = FrameSaver(episode_img_dir)

    cur_obs_list = []
    current_position, current_heading_point = execute_initial_context(
        env, init_positions, init_headings, cur_obs_list, image_size, frame_saver
    )

    floorplan_tensor = floorplan_tensor.to(device)
    goal_pos_t = torch.as_tensor(goal_pos, dtype=torch.float32, device=device).unsqueeze(0)

    save_states = []
    step = 0
    arrived = False
    pending_waypoints = []

    while step < max_steps:
        turned_due_to_collision = False
        collision_this_step = False
        if pending_waypoints:
            current_position, current_heading_point, collision_this_step = move_along_waypoints(
                env, current_position, pending_waypoints, cur_obs_list, context_size, travers_map,
                image_size, save_states, frame_saver,
            )
            pending_waypoints = []
            if collision_this_step:
                logger.info(f"step {step}: collision, attempting recovery turn")
                current_heading_point, still_colliding = recover_from_collision(
                    env, current_position, current_heading_point, cur_obs_list, context_size, travers_map,
                    image_size, save_states, frame_saver,
                )
                turned_due_to_collision = True
                if still_colliding:
                    logger.info(f"step {step}: stuck after recovery attempts, ending episode")
                    break

        cur_pos = current_position[:2]
        cur_heading = current_heading_point[:2]

        dist_to_goal = np.linalg.norm(cur_pos - goal_pos)
        logger.info(f"step {step}: pos=({cur_pos[0]:.2f}, {cur_pos[1]:.2f}) distance_to_goal={dist_to_goal:.3f}")
        if dist_to_goal < arrive_th:
            arrived = True
            break

        shortest_path = shortest_path_waypoints(graph, floorplan_grid_size, cur_pos, goal_pos, len_traj_pred)
        if shortest_path is None:
            logger.info(f"step {step}: no path found from current position to goal, ending episode")
            break

        if not turned_due_to_collision:
            current_heading_point = realign_to_shortest_path(
                env, current_position, current_heading_point, shortest_path, cur_obs_list, image_size, frame_saver
            )
            cur_heading = current_heading_point[:2]

        shortest_actions = compute_local_actions(shortest_path, len_traj_pred, metric_waypoint_spacing, waypoint_spacing)
        shortest_actions_t = torch.as_tensor(shortest_actions, dtype=torch.float32, device=device).unsqueeze(0)

        cur_obs = torch.cat(cur_obs_list, dim=0).to(device)
        with torch.no_grad():
            depth_out = depth_generator(cur_obs, output_latent=True)
        depth_latent = depth_out.latent.reshape(1, -1, depth_out.latent.shape[-2], depth_out.latent.shape[-1])

        cur_pos_t = torch.as_tensor(cur_pos, dtype=torch.float32, device=device).unsqueeze(0)
        cur_heading_t = torch.as_tensor(cur_heading, dtype=torch.float32, device=device).unsqueeze(0)

        local_actions = predict_waypoints(
            model, depth_latent, floorplan_tensor, cur_pos_t, cur_heading_t, goal_pos_t,
            shortest_actions_t, noise_scheduler, len_traj_pred, device,
        )
        local_actions = local_actions[0].cpu().numpy() * metric_waypoint_spacing * waypoint_spacing
        global_actions = data_utils.to_global_coords(local_actions, cur_pos, cur_heading)

        # global_actions[0] is always ~(0,0) in the local frame by construction (see
        # compute_local_actions / GlocDiffDataset._compute_actions, which both express the
        # window relative to its own first point): training's diffusion target for index 0
        # is always "no movement", so it carries no real prediction. Real predicted motion
        # starts at index 1.
        for action in global_actions[1 : 1 + EXECUTE_HORIZON]:
            pending_waypoints.append(np.array([action[0], action[1], height]))

        step += 1

    save_states = np.array(save_states)
    save_trajectory_plot(floorplan_path, save_states, goal_pos, trajectory_plot_path)
    return arrived, save_states


REQUIRED_CONFIG_KEYS = [
    "checkpoint_path", "testdataset", "trav_maps_path", "scene_path", "state_save_dir",
    "test_scenes", "traj_index_range", "context_size", "len_traj_pred", "image_size",
    "metric_waypoint_spacing", "waypoint_spacing", "arrive_th", "max_steps",
    "encoding_size", "floorplan_encoder", "cond_predict_scale", "mha_num_attention_heads",
    "mha_num_attention_layers", "mha_ff_dim_factor", "down_dims", "num_diffusion_iters",
]


def validate_config(config):
    """Fail fast with a clear message before loading the model/scenes, rather than crashing
    partway through (after several minutes of setup) on a typo'd or missing config path."""
    missing_keys = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing_keys:
        raise ValueError(f"config is missing required key(s): {', '.join(missing_keys)}")

    path_checks = {
        "checkpoint_path": os.path.isfile,
        "testdataset": os.path.isdir,
        "trav_maps_path": os.path.isdir,
        "scene_path": os.path.isdir,
    }
    bad_paths = [f"{key}={config[key]!r}" for key, exists in path_checks.items() if not exists(config[key])]
    if bad_paths:
        raise FileNotFoundError(f"config path(s) don't exist: {', '.join(bad_paths)}")


def discover_trajs(testdataset, scene_id, floor, traj_index_range):
    """Auto-discover traj_* folders under testdataset/<scene_id>_<floor>/, sorted by trailing
    number, and slice out the configured index range (matches the old script's range(10, 15))."""
    scene_dir = os.path.join(testdataset, f"{scene_id}_{floor}")
    traj_ids = sorted(
        int(f.split("_")[1]) for f in os.listdir(scene_dir)
        if f.startswith("traj") and os.path.isdir(os.path.join(scene_dir, f))
    )
    lo, hi = traj_index_range
    return [f"traj_{traj_ids[i]}" for i in range(lo, min(hi, len(traj_ids)))]


def execute_navigation_task(config):
    validate_config(config)

    # Each run gets its own timestamped subfolder, so repeated runs don't clutter or
    # overwrite each other's logs/images/trajectories.
    state_save_dir = os.path.join(config["state_save_dir"], time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(state_save_dir, exist_ok=True)

    log_path = setup_logging(state_save_dir)
    logger.info(f"full logs written to {log_path}")
    logger.info(f"this run's outputs are under {state_save_dir}")

    device = torch.device(config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    render_device_idx = device.index if device.type == "cuda" and device.index is not None else 0

    model, noise_scheduler = load_model(config["checkpoint_path"], config, device)
    depth_generator = diffusers_pkg.MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
    ).to(device)

    testdataset = config["testdataset"]
    trav_maps_path = config["trav_maps_path"]
    scene_path_root = config["scene_path"]
    image_size = tuple(config["image_size"])

    headless = config.get("headless", True)
    settings = MeshRendererSettings(enable_shadow=False, msaa=False)

    scenes = config["test_scenes"]
    results = []  # (scene, traj_name, arrived) -- arrived is None if the episode crashed
    for scene_idx, scene in enumerate(scenes, start=1):
        scene_id, floor = scene.split("_")[0], int(scene.split("_")[1])
        logger.info(f"=== scene {scene_idx}/{len(scenes)}: {scene} ===")
        height = get_floor_height(scene_path_root, scene_id, floor)

        # Three different "floorplan" images are in play, at two different pixel scales:
        # - foucused_map.png / map.png / floor_trav_test_*.png (trav_maps_path) are all
        #   1000x1000, 1px == 1cm, matching check_collision()'s scale -- used for
        #   pathfinding and for the trajectory visualization.
        # - floorplan.png (testdataset) is a small, differently-scaled image used only as
        #   the resized tensor fed into deep_floor_net; never use it for pixel math.
        trav_map_dir = scene_id if floor == 0 else scene
        pathfinding_map_path = os.path.join(trav_maps_path, trav_map_dir, "foucused_map.png")
        floorplan_vis_path = os.path.join(trav_maps_path, trav_map_dir, "map.png")
        travers_map_path = os.path.join(trav_maps_path, trav_map_dir, f"floor_trav_test_{floor}_modified_8bit.png")

        graph, floorplan_grid_size = build_graph(pathfinding_map_path)
        travers_map = np.array(Image.open(travers_map_path))

        floorplan_tensor_path = os.path.join(testdataset, scene, "floorplan.png")
        floorplan_tensor = data_utils.resize_and_aspect_crop(
            Image.open(floorplan_tensor_path).convert("RGB"), image_size, data_utils.IMAGE_ASPECT_RATIO
        ).unsqueeze(0)

        s = Simulator(
            mode="headless" if headless else "gui_interactive",
            image_width=512,
            image_height=512,
            rendering_settings=settings,
            device_idx=render_device_idx,
        )
        try:
            igibson_scene = StaticIndoorScene(scene_id, build_graph=True)
            s.import_scene(igibson_scene)
            if not headless:
                s.viewer.initial_pos = config["initial_pos"]
                s.viewer.initial_view_direction = config["initial_view_direction"]
                s.viewer.reset_viewer()

            traj_names = discover_trajs(testdataset, scene_id, floor, config["traj_index_range"])
            for traj_idx, traj_name in enumerate(traj_names, start=1):
                logger.info(f"--- {scene} trajectory {traj_idx}/{len(traj_names)}: {traj_name} ---")
                try:
                    traj_path = os.path.join(testdataset, scene, traj_name, traj_name + ".npy")
                    traj_data = np.load(traj_path)

                    episode_dir = os.path.join(state_save_dir, scene, traj_name)
                    episode_img_dir = os.path.join(episode_dir, "frames")
                    os.makedirs(episode_img_dir, exist_ok=True)
                    trajectory_plot_path = os.path.join(episode_dir, "trajectory.png")

                    arrived, save_states = run_episode(
                        s, model, noise_scheduler, depth_generator, graph, floorplan_grid_size,
                        floorplan_tensor, floorplan_vis_path, travers_map, traj_data, height, config, device,
                        episode_img_dir, trajectory_plot_path,
                    )
                    logger.info(f"{scene}/{traj_name}: {'ARRIVED' if arrived else 'FAILED'}")
                    np.savetxt(os.path.join(episode_dir, "states.txt"), save_states, fmt="%f")
                    results.append((scene, traj_name, arrived))
                except Exception:
                    logger.exception(f"{scene}/{traj_name}: episode crashed, skipping")
                    results.append((scene, traj_name, None))
        finally:
            s.disconnect()

    n_arrived = sum(r[2] is True for r in results)
    n_crashed = sum(r[2] is None for r in results)
    logger.info(
        f"=== done: {n_arrived}/{len(results)} arrived"
        + (f", {n_crashed} crashed" if n_crashed else "")
        + f" -- run compute_nav_metrics.py for SPL/SoftSPL ==="
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GlocDiff closed-loop iGibson rollout")
    parser.add_argument("--config", "-c", default="../config/test_glocdiff.yaml", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    execute_navigation_task(config)
