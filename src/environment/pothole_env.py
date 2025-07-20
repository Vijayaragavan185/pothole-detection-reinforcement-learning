"""
src/environment/pothole_env.py

Custom OpenAI-Gym environment that feeds 5-frame video sequences to an RL
agent which must choose an optimal confidence-threshold action for pothole
detection.  Reward is based on comparison with pixel-level ground-truth masks.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random, json, gc, psutil, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.config import ENV_CONFIG, VIDEO_CONFIG, PATHS


class VideoBasedPotholeEnv(gym.Env):
    """Revolutionary video-based RL environment for pothole detection."""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    # --------------------------------------------------------------------- #
    #                        ░░ INITIALISATION ░░                           #
    # --------------------------------------------------------------------- #
    def __init__(self, split: str = "train", render_mode: str | None = None,
                 max_memory_mb: int = 4096, lazy: bool = True):
        super().__init__()

        self.split = split
        self.render_mode = render_mode
        self.max_memory_mb = max_memory_mb
        self.lazy = lazy

        # Core video parameters ------------------------------------------------
        self.sequence_length = VIDEO_CONFIG["sequence_length"]          # 5
        self.input_size      = VIDEO_CONFIG["input_size"]               # (224,224)

        # Action-space: five confidence thresholds ---------------------------
        self.action_space     = spaces.Discrete(5)
        self.action_thresholds = ENV_CONFIG["action_thresholds"]        # [0.3 … 0.9]

        # Observation-space: (seq_len, H, W, C) in [0,1] ---------------------
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (self.sequence_length,
                     self.input_size[1],   # height
                     self.input_size[0],   # width
                     3),                   # RGB
            dtype = np.float32
        )

        # Rewards -------------------------------------------------------------
        self.reward_correct         = ENV_CONFIG["reward_correct"]          # +10
        self.reward_false_positive  = ENV_CONFIG["reward_false_positive"]   #  -5
        self.reward_missed          = ENV_CONFIG["reward_missed"]           # -20

        # Episode buffers ----------------------------------------------------
        self.episode_sequences      = []
        self.episode_ground_truths  = []
        self.episode_metadata       = []

        # Runtime state ------------------------------------------------------
        self.current_sequence       = None
        self.current_ground_truth   = None
        self.sequence_index         = 0
        self.episode_rewards        = []
        self.episode_actions        = []
        self.episode_count          = 0

        # Performance counters ----------------------------------------------
        self.total_correct_detections = 0
        self.total_false_positives    = 0
        self.total_missed_detections  = 0
        self.failed_loads             = 0
        self.successful_loads         = 0

        # Load dataset -------------------------------------------------------
        self._load_dataset()

        print("✅ RL Environment ready:",
              f"{len(self.episode_sequences):,} sequences loaded for split='{split}'.",
              f"Memory limit = {max_memory_mb} MB")

    # --------------------------------------------------------------------- #
    #                        ░░ DATA LOADING ░░                             #
    # --------------------------------------------------------------------- #
    def _get_mem_usage_mb(self) -> float:
        return psutil.Process().memory_info().rss / 1024 / 1024

    def _safe_np_load(self, file: Path, expected_frames: int,
                      mmap: str | None = "r") -> np.ndarray | None:
        """Robust NumPy loader with size/shape sanity checks."""
        try:
            if not file.exists():
                return None
            if file.stat().st_size / (1024*1024) > 50:          # >50 MB file
                return None
            if self._get_mem_usage_mb() > self.max_memory_mb:
                gc.collect()
                if self._get_mem_usage_mb() > self.max_memory_mb:
                    return None

            arr = np.load(file, mmap_mode=mmap)
            if len(arr.shape) not in {3,4}:                     # (F,H,W[,C])
                return None
            if arr.shape[0] != expected_frames:
                return None
            arr = np.asarray(arr, dtype=np.float32)
            if np.isnan(arr).any() or np.isinf(arr).any():
                return None
            if arr.max() > 1.0 or arr.min() < 0.0:
                arr = np.clip(arr, 0.0, 1.0)
            return arr
        except Exception:
            return None

    def _load_sequences_from_dir(self, seq_dir: Path,
                                 gt_root: Path,
                                 data_type: str):
        for video_dir in [d for d in seq_dir.iterdir() if d.is_dir()]:
            video_name = video_dir.name.split('_aug')[0]
            seq_files  = sorted((video_dir / "sequences").glob("*.npy"))

            # Cap per-video load to avoid memory blow-up
            for seq_file in seq_files[:50]:
                seq = self._safe_np_load(seq_file, self.sequence_length)
                if seq is None:
                    self.failed_loads += 1
                    continue

                # Ground-truth -------------------------------------------------
                gt = None
                gt_dir = gt_root / video_name / "sequences"
                if gt_dir.exists():
                    seq_idx = seq_file.stem.split('_')[-1].split('aug')[0]
                    gt_file = gt_dir / f"mask_sequence_{seq_idx}.npy"
                    gt      = self._safe_np_load(gt_file, self.sequence_length)

                # Metadata ----------------------------------------------------
                meta = {"video_name": video_name,
                        "sequence_idx": seq_file.stem.split('_')[-1],
                        "data_type": data_type,
                        "has_ground_truth": gt is not None}

                self.episode_sequences.append(seq)
                self.episode_ground_truths.append(gt)
                self.episode_metadata.append(meta)
                self.successful_loads += 1

                if len(self.episode_sequences) >= 5_000:
                    return  # memory safety cap

    def _create_fallback_data(self):
        """Small synthetic set if zero real sequences found."""
        for i in range(10):
            seq = np.random.rand(self.sequence_length, 224, 224, 3).astype(np.float32)
            mask = np.zeros((self.sequence_length, 224, 224), dtype=np.float32)
            if random.random() > 0.5:
                mask[:, 80:150, 80:150] = 1.0
            self.episode_sequences.append(seq)
            self.episode_ground_truths.append(mask)
            self.episode_metadata.append({"video_name": f"fallback_{i:03d}",
                                          "sequence_idx": "0",
                                          "data_type": "fallback",
                                          "has_ground_truth": True})

    def _load_dataset(self):
        print(f"📊 Loading '{self.split}' dataset …")
        root_frames = PATHS["processed_frames"]
        root_gt     = PATHS["ground_truth"] / self.split

        self._load_sequences_from_dir(root_frames / f"{self.split}_optimized",
                                      root_gt, "optimized")
        self._load_sequences_from_dir(root_frames / f"{self.split}_augmented",
                                      root_gt, "augmented")

        if not self.episode_sequences:
            print("⚠️ No sequences found – generating fallback data.")
            self._create_fallback_data()

        gc.collect()
        print(f"🚀 Dataset ready: {len(self.episode_sequences):,} sequences "
              f"(success={self.successful_loads}, skipped={self.failed_loads})")

    # --------------------------------------------------------------------- #
    #                        ░░ CORE ENV API ░░                             #
    # --------------------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = random.randrange(len(self.episode_sequences))
        self.current_sequence     = self.episode_sequences[idx]
        self.current_ground_truth = self.episode_ground_truths[idx]
        self.current_metadata     = self.episode_metadata[idx]

        self.sequence_index = 0
        self.episode_rewards.clear()
        self.episode_actions.clear()
        self.episode_count += 1

        observation = self.current_sequence.astype(np.float32)
        info = {"episode": self.episode_count,
                "metadata": self.current_metadata,
                "has_ground_truth": self.current_ground_truth is not None,
                "memory_usage_mb": round(self._get_mem_usage_mb(), 1)}
        return observation, info

    def _ground_truth_has_pothole(self) -> bool:
        if self.current_ground_truth is None:
            return False
        ratio = (self.current_ground_truth > 0).sum() / self.current_ground_truth.size
        return ratio > 0.01

    def _simulate_detection_confidence(self) -> float:
        if self.current_ground_truth is None:
            return random.uniform(0.25, 0.75)

        if self._ground_truth_has_pothole():
            base = random.uniform(0.65, 0.88)
        else:
            base = random.uniform(0.15, 0.42)
        return np.clip(base + random.uniform(-0.12, 0.12), 0.0, 1.0)

    def _calculate_reward(self, detected: bool) -> int:
        if self.current_ground_truth is None:
            return 0

        has_pothole = self._ground_truth_has_pothole()

        if has_pothole and detected:
            self.total_correct_detections += 1
            return self.reward_correct
        if (not has_pothole) and (not detected):
            self.total_correct_detections += 1
            return self.reward_correct
        if (not has_pothole) and detected:
            self.total_false_positives += 1
            return self.reward_false_positive
        self.total_missed_detections += 1
        return self.reward_missed

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        threshold           = self.action_thresholds[action]
        confidence          = self._simulate_detection_confidence()
        agent_detects       = confidence > threshold
        reward              = self._calculate_reward(agent_detects)

        self.episode_rewards.append(reward)
        self.episode_actions.append(action)

        info = {"action": action,
                "confidence_threshold": threshold,
                "detection_confidence": confidence,
                "agent_decision": agent_detects,
                "reward": reward,
                "episode_total_reward": sum(self.episode_rewards),
                "ground_truth_has_pothole": self._ground_truth_has_pothole(),
                "metadata": self.current_metadata,
                "memory_usage_mb": round(self._get_mem_usage_mb(), 1)}

        obs = self.current_sequence.astype(np.float32)
        done = True
        truncated = False
        return obs, reward, done, truncated, info

    # --------------------------------------------------------------------- #
    #                        ░░ VISUAL RENDER ░░                            #
    # --------------------------------------------------------------------- #
    def render(self):
        if self.render_mode is None or self.current_sequence is None:
            return None
        frame = (self.current_sequence[0] * 255).astype(np.uint8)
        if self.render_mode == "human":
            import cv2
            cv2.imshow("Pothole-RL Frame", frame)
            cv2.waitKey(1)
        elif self.render_mode == "rgb_array":
            return frame

    def close(self):
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.episode_sequences.clear()
        self.episode_ground_truths.clear()
        self.episode_metadata.clear()
        gc.collect()

    # --------------------------------------------------------------------- #
    #                        ░░ METRICS HELPERS ░░                          #
    # --------------------------------------------------------------------- #
    def get_performance_stats(self) -> dict:
        total = (self.total_correct_detections +
                 self.total_false_positives +
                 self.total_missed_detections)
        if total == 0:
            return {"message": "No decisions made yet"}

        precision = (self.total_correct_detections /
                     (self.total_correct_detections +
                      self.total_false_positives)
                     * 100 if self.total_correct_detections else 0)
        recall = (self.total_correct_detections /
                  (self.total_correct_detections +
                   self.total_missed_detections)
                  * 100 if self.total_correct_detections else 0)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0)

        return {
            "total_episodes": self.episode_count,
            "total_decisions": total,
            "correct_detections": self.total_correct_detections,
            "false_positives": self.total_false_positives,
            "missed_detections": self.total_missed_detections,
            "accuracy": round(self.total_correct_detections / total * 100, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1, 2),
            "average_episode_reward":
                np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "memory_usage_mb": round(self._get_mem_usage_mb(), 1)
        }

    def get_dataset_info(self) -> dict:
        return {
            "split": self.split,
            "total_sequences": len(self.episode_sequences),
            "successful_loads": self.successful_loads,
            "failed_loads": self.failed_loads,
            "has_ground_truth":
                sum(gt is not None for gt in self.episode_ground_truths),
            "memory_limit_mb": self.max_memory_mb,
            "current_memory_mb": round(self._get_mem_usage_mb(), 1)
        }


# ------------------------------------------------------------------------- #
#                        ░░ QUICK SELF-TEST ░░                              #
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    env = VideoBasedPotholeEnv(split="train", max_memory_mb=4096, lazy=True, verbose=True)
    obs, info = env.reset()
    print("Observation shape:", obs.shape, "| Episode info keys:", list(info.keys()))
    for a in range(env.action_space.n):
        obs, reward, done, trunc, info = env.step(a)
        print(f"Action {a} → reward {reward:+3}")
    print("Stats:", env.get_performance_stats())
    env.close()
