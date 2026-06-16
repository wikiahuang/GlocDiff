"""
a train data sample is like:
an RGB image:
a shortest path: 20 * 2, consits of future 20 steps in meters in local coordinate system;
a collision-free action: 20 * 2, consits of future 20 steps in meters in local coordinate system;
"""
import os

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

from utils import data_utils

class GlocDiffDataset(Dataset):
    def __init__(
        self, 
        data_folder, 
        scene_names,
        image_size,
        waypoint_spacing,      
        len_traj_pred, 
        context_size,
        end_slack,   
        normalize,
        obs_type,
        trav_map_path,
        ):
        """
        Args:
            data_folder (str): root folder containing per-scene trajectory data
            scene_names (list): scene names to include in this split
            image_size (Tuple[int, int]): size to resize observation images to
            waypoint_spacing (int): number of frames between consecutive waypoints
            len_traj_pred (int): number of future waypoints to predict
            context_size (int): number of past observation frames per sample
            end_slack (int): number of frames to ignore at the end of each trajectory
            normalize (bool): whether to normalize actions to waypoint units
            obs_type (str): observation modality ("image")
            trav_map_path (str): root folder containing per-scene traversable maps
        """
        self.data_folder = data_folder
        self.scene_names = scene_names
        self.context_size = context_size
        self.waypoint_spacing = waypoint_spacing
        self.end_slack = end_slack  
        self.len_traj_pred = len_traj_pred
        self.traj_names = {}
        self.trajectory_cache = {}
        self.shortest_path_cache = {}
        self.metric_waypoint_spacing = 0.045
        self.image_size = image_size   
        self.learn_angle = False     
        self.normalize = normalize
        self.trav_map_path = trav_map_path  
        for scene_name in scene_names:
            self.traj_names[scene_name] = []
            for file in os.listdir(os.path.join(data_folder, scene_name)):
                if os.path.isdir(os.path.join(data_folder, scene_name, file)) and file.startswith("traj"):
                    self.traj_names[scene_name].append(file)
            self.traj_names[scene_name].sort()
        self._load_index()
        self.num_action_params = 2

    def _load_index(self):
        """
        Load the cached sample index from disk, building and caching it if missing.
        """
        index_path = os.path.join(self.data_folder, f"dataset_n{self.context_size}_slack_{self.end_slack}.npz")
        data_split = self.data_folder.split('/')[-1]
        try:
            npz = np.load(index_path)
            self.data_index = npz['data_index']
            print(data_split + ' index loaded, size:', len(self.data_index))  
        except:
            print('Building index...')
            self.data_index =np.array(self._build_index()) 
            np.savez(index_path, data_index=self.data_index)
            print(data_split + ' index built, size:', len(self.data_index))
    def _build_caches(self):
        """
        Placeholder for precomputing on-disk caches. Currently a no-op.
        """
        pass

    def _get_shortest_path(self, scene_name, traj_name, current_time):
        """
        Load (and cache) the precomputed shortest path for a given frame.

        Args:
            scene_name (str): scene name
            traj_name (str): trajectory name
            current_time (int): frame index
        Returns:
            np.ndarray: shortest path waypoints
        """
        k = scene_name + '-' + traj_name + '-' + str(current_time)
        if k in self.shortest_path_cache:
            return self.shortest_path_cache[k]
        else:
            with open(os.path.join(self.data_folder, scene_name, traj_name, "shortest_paths", "shortest_path_for_{:05d}.npy".format(current_time)), "rb") as f:
                shortest_path = np.load(f)
            self.shortest_path_cache[k] = shortest_path
            return shortest_path
    
    def _build_index(self, use_tqdm: bool = False):
        """
        Scan every trajectory's shortest-path files to enumerate valid (scene/traj, frame) samples.

        Args:
            use_tqdm (bool): whether to show a progress bar while scanning scenes
        Returns:
            list: list of (scene_name-traj_name, frame_time) tuples
        """
        samples_index = []
        for scene_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            trajs = self.traj_names[scene_name]
            for traj_name in trajs:
                shortest_path_dir = os.path.join(self.data_folder, scene_name, traj_name, "shortest_paths")
                if not os.path.exists(shortest_path_dir):
                    continue   
                files = os.listdir(shortest_path_dir)    
                npy_files = [f for f in files if f.endswith('.npy')]
                time_list = [(f.split('_')[-1].split('.')[0]) for f in npy_files]
                scene_name_traj_name = scene_name + '-' + traj_name    
                for current_time in time_list:
                    samples_index.append((scene_name_traj_name, current_time))           
                # traj_data_path = os.path.join(self.data_folder, scene_name, traj_name, traj_name + ".npy")
                # traj_data = np.load(traj_data_path) 
                # traj_len = traj_data.shape[0]
                # begin_time = self.context_size * self.waypoint_spacing + 1
                # end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
                # for curr_time in range(begin_time, end_time, 5):    # is the whole seq len is less than pred_len begin_time will be greater than end_time and the loop will not run               
                #     samples_index.append((scene_name_traj_name, curr_time))
        return samples_index
    
    def _compute_actions(self, traj_data):
        """
        Convert a window of raw (position, heading) trajectory data into local-frame actions.

        Args:
            traj_data (np.ndarray): trajectory window, shape (len_traj_pred, 4) with columns [x, y, heading_x, heading_y]
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: actions, current position, current heading, current position in local coordinates
        """
        yaw = traj_data[:, 2:].copy()
        positions = traj_data[:, :2].copy()

        cur_pos = positions[0]
        cur_ori = yaw[0]
        cur_ori = cur_pos + (cur_ori - cur_pos) / np.linalg.norm(cur_ori - cur_pos)
        waypoints = data_utils.to_local_coords(positions, positions[0], yaw[0])
        cur_pos_local = data_utils.to_local_coords(cur_pos, positions[0], yaw[0])
        
        assert waypoints.shape == (self.len_traj_pred , 2), f"{waypoints.shape} and {(self.len_traj_pred, 2)} should be equal"

        if self.learn_angle:
            yaw = yaw[1:] - yaw[0]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            actions = waypoints
        
        if self.normalize:
            actions[:, :2] /= self.metric_waypoint_spacing * self.waypoint_spacing  # transform meters to waypoints(steps)
            cur_pos /= self.metric_waypoint_spacing * self.waypoint_spacing 
            cur_ori /= self.metric_waypoint_spacing * self.waypoint_spacing

        assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, cur_pos, cur_ori, cur_pos_local
    
    def _get_trajectory(self, scene_name, trajectory_name, current_time):
        """
        Load (and cache) a trajectory's full waypoint array and slice out the future window.

        Args:
            scene_name (str): scene name
            trajectory_name (str): trajectory name
            current_time (int): starting frame index
        Returns:
            np.ndarray: waypoints from current_time through the prediction horizon
        """
        k = scene_name+'-'+trajectory_name
        if k in self.trajectory_cache:
            return self.trajectory_cache[k][current_time:current_time+self.len_traj_pred * self.waypoint_spacing]
        else:
            with open(os.path.join(self.data_folder, scene_name, trajectory_name, trajectory_name + ".npy"), "rb") as f:
                traj_data = np.load(f)           
            self.trajectory_cache[k] = traj_data
            traj_data = traj_data[current_time:current_time+self.len_traj_pred * self.waypoint_spacing]
            return traj_data
        
    def _get_metric_goal(self, scene_name, trajectory_name):
        """
        Load (and cache) a trajectory's full waypoint array and return its final pose as the goal.

        Args:
            scene_name (str): scene name
            trajectory_name (str): trajectory name
        Returns:
            np.ndarray: final (position, heading) of the trajectory
        """
        k = scene_name+'-'+trajectory_name
        if k in self.trajectory_cache:
            return self.trajectory_cache[k][-1]   
        else:
            with open(os.path.join(self.data_folder, scene_name, trajectory_name, trajectory_name + ".npy"), "rb") as f:
                traj_data = np.load(f)           
            self.trajectory_cache[k] = traj_data
            metric_goal = traj_data[-1]
            return metric_goal
    
    def _load_image(self, scene_name, traj, name):
        """
        Load and resize a single image (an observation frame or the scene's floor plan).

        Args:
            scene_name (str): scene name
            traj (str): trajectory name (ignored when name == "floorplan")
            name (int, str): frame index, or "floorplan" to load the scene's floor plan image
        Returns:
            torch.Tensor: resized image as a tensor
        """
        if name == "floorplan":
            image_path = data_utils.get_data_path(self.data_folder, scene_name, name)
        else:
            image_path = data_utils.get_data_path(os.path.join(self.data_folder, scene_name), traj, name)
        
        try:   # directedly load from disk
            # time0 = time.time()
            with open(image_path, "rb") as f:
                result = data_utils.img_path_to_data(f, self.image_size)
            # time1 = time.time()
            # print(f"get image time: {time1 - time0}")
            return result
            
        except TypeError:
            print(f"Failed to load image {image_path}")
    def __len__(self):
        """
        Returns:
            int: number of samples in the dataset
        """
        return len(self.data_index)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): sample index
        Returns:
            Tuple: (sample id, observation images, shortest-path actions, collision-free
            actions, collision-free waypoints in meters, floor plan image, metric goal
            position, traversable map path)
        """
        scene_traj, current_time = self.data_index[idx]
        scene, traj = scene_traj.split('-')
        # load image
        context_times =  list(
            range(
                int(current_time) + -self.context_size * self.waypoint_spacing + 1,
                int(current_time) + 1,
                self.waypoint_spacing,
            )
        )
        context = [(traj, name) for name in context_times]
        obs_image = torch.cat([
            self._load_image(scene, traj, name) for traj, name in context
        ])
        #load floor plan
        floor_plan = self._load_image(scene, traj, 'floorplan')
        
        # load shortest path
        shortest_path = self._get_shortest_path(scene, traj, int(current_time))
        shortest_actions, _, _, _ = self._compute_actions(shortest_path) 
        
        #load goal location
        metric_goal = self._get_metric_goal(scene, traj)[:2]
        # load collision-free actions
        meter_collision_free_actions = self._get_trajectory(scene, traj, int(current_time))
        collision_free_actions, _, _, _ = self._compute_actions(meter_collision_free_actions) 
        
        
        #load traversable map path
        if scene.endswith('0'):
            traversable_map_path = os.path.join(self.trav_map_path, scene.split('_')[0], 'floor_trav_test_0_modified_8bit.png')
        else:
            traversable_map_path = os.path.join(self.trav_map_path, scene, 'floor_trav_test_1_modified_8bit.png')
        
        return (
            scene_traj+'-'+str(current_time),
            torch.as_tensor(obs_image, dtype=torch.float32), 
            torch.as_tensor(shortest_actions, dtype=torch.float32),
            torch.as_tensor(collision_free_actions, dtype=torch.float32),
            torch.as_tensor(meter_collision_free_actions, dtype=torch.float32),
            torch.as_tensor(floor_plan, dtype=torch.float32),
            torch.as_tensor(metric_goal, dtype=torch.float32),
            traversable_map_path
                )