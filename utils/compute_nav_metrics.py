"""
Navigation metrics (success rate, SPL, soft-SPL, collisions) from the per-episode
trajectory logs that test_glocdiff.py saves to state_save_dir/{scene}_{traj_name}.txt.

Each log row is [x, y, heading_x, heading_y, collision] (see save_states in
test_glocdiff.py:run_episode). The episode's goal and shortest-path length come from
the ground-truth trajectory file under testdataset/{scene}/{traj_name}/{traj_name}.npy.
"""
import argparse
import os
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class EpisodeResult:
    arrived: bool
    collisions: int
    shortest_path_length: float
    traveled_length: float
    initial_distance: float
    final_distance: float

    @property
    def spl(self) -> float:
        """Success weighted by Path Length."""
        denom = max(self.traveled_length, self.shortest_path_length)
        return self.arrived * self.shortest_path_length / denom if denom > 0 else float(self.arrived)

    @property
    def soft_spl(self) -> float:
        """SPL weighted by goal-progress instead of a hard arrival flag."""
        progress = 1.0 - self.final_distance / self.initial_distance if self.initial_distance > 0 else 0.0
        denom = max(self.traveled_length, self.shortest_path_length)
        return progress * self.shortest_path_length / denom if denom > 0 else progress


def path_length(positions: np.ndarray) -> float:
    """Total length (meters) of a sequence of (x, y) positions."""
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def evaluate_episode(states: np.ndarray, gt_trajectory: np.ndarray, arrive_th: float,
                      collision_limit: int) -> EpisodeResult:
    """
    Replay a saved trajectory log against the ground-truth episode's goal.

    Arrival uses the same criterion as test_glocdiff.py (distance to goal < arrive_th).
    The episode is cut short, and counted as a failure, once `collisions` collided
    steps have accumulated -- this bounds how much a single stuck/colliding rollout
    can inflate the traveled-distance-based metrics below.
    """
    positions = states[:, :2]
    collided = states[:, 4].astype(bool)
    goal = gt_trajectory[-1, :2]

    initial_distance = float(np.linalg.norm(positions[0] - goal))
    final_distance = initial_distance
    arrived = False
    collisions = 0
    traveled = 0.0

    for i, pos in enumerate(positions):
        if i > 0:
            traveled += float(np.linalg.norm(pos - positions[i - 1]))
        collisions += int(collided[i])
        final_distance = float(np.linalg.norm(pos - goal))
        if collisions >= collision_limit:
            break
        if final_distance < arrive_th:
            arrived = True
            break

    return EpisodeResult(
        arrived=arrived,
        collisions=collisions,
        shortest_path_length=path_length(gt_trajectory[:, :2]),
        traveled_length=traveled,
        initial_distance=initial_distance,
        final_distance=final_distance,
    )


def resolve_run_dir(state_save_dir: str) -> str:
    """test_glocdiff.py saves each run under state_save_dir/run_<timestamp>/. If state_save_dir
    itself holds such run_* subfolders (rather than episode logs directly), resolve to the most
    recent one; otherwise assume it's already a specific run's folder."""
    run_dirs = sorted(
        d for d in os.listdir(state_save_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(state_save_dir, d))
    )
    if not run_dirs:
        return state_save_dir
    return os.path.join(state_save_dir, run_dirs[-1])


def load_episodes(state_save_dir: str, testdataset: str) -> list:
    """Load every saved episode log paired with its ground-truth trajectory. Episodes are laid
    out as state_save_dir/{scene}/{traj_name}/states.txt (see test_glocdiff.py)."""
    episodes = []
    for scene in sorted(os.listdir(state_save_dir)):
        scene_dir = os.path.join(state_save_dir, scene)
        if not os.path.isdir(scene_dir):
            continue
        for traj_name in sorted(os.listdir(scene_dir)):
            states_path = os.path.join(scene_dir, traj_name, "states.txt")
            if not os.path.isfile(states_path):
                continue
            try:
                states = np.loadtxt(states_path)
                gt_path = os.path.join(testdataset, scene, traj_name, f"{traj_name}.npy")
                gt_trajectory = np.load(gt_path)
            except (OSError, ValueError) as e:
                print(f"skipping {scene}/{traj_name}: {e}")
                continue
            episodes.append((f"{scene}/{traj_name}", states, gt_trajectory))
    return episodes


def summarize(episodes: list, arrive_th: float, collision_limit: int) -> dict:
    results = [evaluate_episode(states, gt, arrive_th, collision_limit) for _, states, gt in episodes]
    return {
        "n_episodes": len(results),
        "success_rate": np.mean([r.arrived for r in results]),
        "spl": np.mean([r.spl for r in results]),
        "soft_spl": np.mean([r.soft_spl for r in results]),
        "mean_collisions": np.mean([r.collisions for r in results]),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute SR / SPL / SoftSPL from saved GlocDiff rollouts")
    parser.add_argument("--config", "-c", default="../config/test_glocdiff.yaml",
                         help="test_glocdiff.yaml; supplies state_save_dir/testdataset/arrive_th defaults")
    parser.add_argument("--state-save-dir", default=None, help="override state_save_dir from --config")
    parser.add_argument("--testdataset", default=None, help="override testdataset from --config")
    parser.add_argument("--arrive-th", type=float, nargs="+", default=None,
                         help="success-distance threshold(s) in meters to evaluate (sweeps if multiple)")
    parser.add_argument("--collision-limit", type=int, nargs="+", default=[10**9],
                         help="episode is cut short and failed after this many collisions (sweeps if multiple)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    state_save_dir = resolve_run_dir(args.state_save_dir or config["state_save_dir"])
    testdataset = args.testdataset or config["testdataset"]
    arrive_thresholds = args.arrive_th or [config["arrive_th"]]

    episodes = load_episodes(state_save_dir, testdataset)
    if not episodes:
        raise SystemExit(f"no episode logs found in {state_save_dir}")
    print(f"evaluating {len(episodes)} episodes from {state_save_dir}\n")

    header = f"{'arrive_th':>10} {'collision_limit':>15} {'SR':>8} {'SPL':>8} {'SoftSPL':>8} {'mean_coll':>10}"
    print(header)
    print("-" * len(header))
    for arrive_th in arrive_thresholds:
        for collision_limit in args.collision_limit:
            m = summarize(episodes, arrive_th, collision_limit)
            print(f"{arrive_th:>10.2f} {collision_limit:>15} {m['success_rate']:>8.3f} "
                  f"{m['spl']:>8.3f} {m['soft_spl']:>8.3f} {m['mean_collisions']:>10.2f}")


if __name__ == "__main__":
    main()
