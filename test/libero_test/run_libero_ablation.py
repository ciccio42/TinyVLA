"""
run_libero_ablation.py  (TinyVLA version)

Ablation study for LIBERO tasks with keyword-only commands.
Supports multiple tasks via --ablation_task_id parameter.
Automatically loads custom BDDL files if available.

Adapted from the OpenVLA version to use TinyVLA's LlavaPythia policy.
"""

import sys
import os
import logging
import json
import time
import pickle
import cv2
import numpy as np
import torch
import tqdm

from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from torchvision import transforms

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DEVICE"] = "cuda"
os.environ["WANDB_DISABLED"] = "true"

# ── TinyVLA / LlavaPythia imports ────────────────────────────────────────────
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
from llava_pythia.conversation import conv_templates, SeparatorStyle
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_pythia.model import *
import utils.torch_utils as TorchUtils

import draccus
import wandb

from libero.libero import benchmark

from utils.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from utils.robot_utils import DATE_TIME, set_seed_everywhere

# ─────────────────────────────────────────────────────────────────────────────

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ── Task suite constants ──────────────────────────────────────────────────────
class TaskSuite(str, Enum):
    LIBERO_GOAL = "libero_goal"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_GOAL: 300,
}


# ── Ablation task configurations ──────────────────────────────────────────────

def get_ablation_tasks(task_id: int) -> dict:
    """Return ablation configuration for the given 0-indexed task ID."""

    ABLATION_CONFIGS = {
        # Task 8: "Turn on the stove" (task_id=7, 0-indexed)
        7: {
            "task_name": "Turn on the stove",
            "tests": {
                "stove1": {
                    "bddl_file": "turn_on_the_stove_ablation_stove1.bddl",
                    "expected_command": "stove",
                },
                "stove2": {
                    "bddl_file": "turn_on_the_stove_ablation_stove2.bddl",
                    "expected_command": "bowl stove",
                },
                "stove3": {
                    "bddl_file": "turn_on_the_stove_ablation_stove3.bddl",
                    "expected_command": "plate stove",
                },
                "stove4": {
                    "bddl_file": "turn_on_the_stove_ablation_stove4.bddl",
                    "expected_command": "Turn on",
                },
            },
        },
        # Task 9: "Put the bowl on the plate" (task_id=8, 0-indexed)
        8: {
            "task_name": "Put the bowl on the plate",
            "tests": {
                "bowl_plate1": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate1.bddl",
                    "expected_command": "bowl",
                },
                "bowl_plate2": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate2.bddl",
                    "expected_command": "plate",
                },
                "bowl_plate3": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate3.bddl",
                    "expected_command": "bowl plate",
                },
                "bowl_plate4": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate4.bddl",
                    "expected_command": "Put plate",
                },
                "bowl_plate5": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate5.bddl",
                    "expected_command": "plate bowl",
                },
            },
        },
    }

    if task_id not in ABLATION_CONFIGS:
        raise ValueError(
            f"No ablation configuration found for task_id={task_id}. "
            f"Available task IDs: {list(ABLATION_CONFIGS.keys())}"
        )

    return ABLATION_CONFIGS[task_id]


# ── Configuration dataclass ───────────────────────────────────────────────────

@dataclass
class GenerateConfig:
    # fmt: off

    ###########################################################################
    # Ablation-specific parameters
    ###########################################################################
    ablation_task_id: int = 7  # 0-indexed task ID to ablate (default=7 → Task 8)
    ablation_test_key: str = ""  # If set, run only this variant (e.g. "stove1"); empty = run all

    ###########################################################################
    # Model-specific parameters (TinyVLA)
    ###########################################################################
    model_path: str = ""        # Path to the TinyVLA LoRA checkpoint
    model_base: str = ""        # Path to the base LlavaPythia model
    model_family: str = "tiny_vla"

    ###########################################################################
    # LIBERO environment-specific parameters
    ###########################################################################
    task_suite_name: str = TaskSuite.LIBERO_GOAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    ###########################################################################
    # Utils
    ###########################################################################
    run_id_note: Optional[str] = None
    local_log_dir: str = "/mnt/beegfs/a.cardamone7/outputs/logs"

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    seed: int = 42
    run_number: int = 0

    debug: bool = False
    # fmt: on


# ── Validation ────────────────────────────────────────────────────────────────

def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.model_path, "model_path must not be empty!"
    assert cfg.model_base, "model_base must not be empty!"
    assert cfg.task_suite_name == TaskSuite.LIBERO_GOAL, (
        "Ablation only works with libero_goal!"
    )
    # Validate that ablation config exists
    get_ablation_tasks(cfg.ablation_task_id)


# ── Policy class (identical to run_libero_eval.py) ───────────────────────────

class llava_pythia_act_policy:
    """TinyVLA LlavaPythia policy wrapper."""

    def __init__(self, policy_config, data_args=None):
        super(llava_pythia_act_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        model_base = policy_config["model_base"] if policy_config["enable_lora"] else None
        model_name = get_model_name_from_path(policy_config["model_path"])
        model_path = policy_config["model_path"]

        self.tokenizer, self.policy, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name, False, False
        )
        self.config = LlavaPythiaConfig.from_pretrained(
            "/".join(model_path.split("/")[:-1]), trust_remote_code=True
        )

    def process_batch_to_llava(self, curr_image, robo_state, raw_lang):
        """Processes a batch of observations for LlavaPythia model input."""
        self.conv = conv_templates[self.policy_config["conv_mode"]].copy()

        if len(curr_image.shape) == 5:
            curr_image = curr_image.squeeze(0)

        image, image_r = torch.chunk(curr_image, 2, dim=0)

        image = self.expand2square(image, tuple(x for x in self.image_processor.image_mean))
        image_tensor = self.image_processor.preprocess(
            image, return_tensors="pt", do_normalize=True, do_rescale=False, do_center_crop=False
        )["pixel_values"]
        image_tensor = image_tensor.to(self.policy.device, dtype=self.policy.dtype)

        image_r = self.expand2square(image_r, tuple(x for x in self.image_processor.image_mean))
        image_tensor_r = self.image_processor.preprocess(
            image_r, return_tensors="pt", do_normalize=True, do_rescale=False, do_center_crop=False
        )["pixel_values"]
        image_tensor_r = image_tensor_r.to(self.policy.device, dtype=self.policy.dtype)

        inp = raw_lang
        assert image is not None, "image must be provided."

        if self.policy.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
        self.conv.append_message(self.conv.roles[0], inp)
        image = None

        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        prompt += " <|endoftext|>"

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        states = robo_state.to(self.policy.device, dtype=self.policy.dtype)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attn_mask,
            images=image_tensor,
            images_r=image_tensor_r,
            states=states.unsqueeze(0),
        )
        return data_dict

    def expand2square(self, pil_imgs, background_color):
        batch_size, channels, height, width = pil_imgs.shape
        max_dim = max(height, width)
        expanded_imgs = np.full(
            (batch_size, max_dim, max_dim, channels), background_color, dtype=np.float32
        )

        if height == width:
            expanded_imgs = pil_imgs.permute(0, 2, 3, 1).cpu().numpy()
        elif height > width:
            offset = (max_dim - width) // 2
            expanded_imgs[:, :height, offset : offset + width, :] = pil_imgs.permute(0, 2, 3, 1).cpu().numpy()
        else:
            offset = (max_dim - height) // 2
            expanded_imgs[:, offset : offset + height, :width, :] = pil_imgs.permute(0, 2, 3, 1).cpu().numpy()

        expanded_imgs = torch.tensor(expanded_imgs).to(dtype=pil_imgs.dtype, device=pil_imgs.device)
        return expanded_imgs


# ── Observation helpers ───────────────────────────────────────────────────────

def convert_actions(pred_action):
    """Convert rot6d format to euler angles."""
    cur_xyz = pred_action[:3]
    cur_rot6d = pred_action[3:9]
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)

    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)
    cur_euler = TorchUtils.rot_6d_to_euler_angles(rot_6d=cur_rot6d, convention="XYZ").squeeze().numpy()
    pred_action = np.concatenate((cur_xyz, cur_euler, cur_gripper))
    return pred_action


def get_obs(obs, stats):
    """Extract and normalise images + proprioceptive state from the environment."""
    images = np.array([
        cv2.resize(obs["agentview_image"][::-1, ::-1], (320, 180)),
        cv2.resize(obs["robot0_eye_in_hand_image"][::-1, ::-1], (320, 180)),
    ])
    states = np.concatenate((
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ))
    states = (states - stats["qpos_mean"]) / stats["qpos_std"]
    return images, states


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def setup_logging(cfg: GenerateConfig, task_name: str):
    """Set up log file and optionally W&B."""
    safe_task_name = task_name.replace(" ", "_").lower()
    run_id = f"ABLATION-Task{cfg.ablation_task_id + 1}-{safe_task_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


# ── Initial-states loading ────────────────────────────────────────────────────

def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    initial_states = task_suite.get_task_init_states(task_id)

    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    policy,
    policy_config,
    initial_state=None,
    log_file=None,
):
    """Run a single rollout episode.  Returns (success, replay_traj)."""
    env.reset()
    to_tensor = transforms.ToTensor()

    # Set initial state
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.reset()
        # Basket position fix (TinyVLA-specific)
        if "basket_1_pos" in obs.keys():
            basket_pos = [0.005, 0.261, 0.035]
            basket_quat = [0.000, 0.000, 0.000, 1.000]
            env.sim.data.set_joint_qpos(
                env.env.objects_dict["basket_1"].joints[0],
                np.concatenate((basket_pos, basket_quat)),
            )
            t = 0
            while t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1

    # ── Policy configuration ──────────────────────────────────────────────
    if policy_config["action_head"] == "act":
        rand_crop_resize = False
        temporal_agg = True
    else:
        rand_crop_resize = True
        temporal_agg = True

    action_dim = policy.config.action_dim
    policy.policy.eval()

    # Load normalisation stats
    stats_path = os.path.join(
        "/".join(policy_config["model_path"].split("/")[:-1]), "dataset_stats.pkl"
    )
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    # Action post-processing
    if policy_config["action_head"] == "act":
        post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
    elif policy_config["action_head"] in ("transformer_diffusion", "droid_diffusion"):
        post_process = (
            lambda a: ((a + 1) / 2) * (stats["action_max"] - stats["action_min"])
            + stats["action_min"]
        )
    else:
        raise NotImplementedError(f"Unknown action_head: {policy_config['action_head']}")

    # Temporal aggregation setup
    query_frequency = policy.config.chunk_size / 2
    if temporal_agg:
        query_frequency = 1
        num_queries = policy.config.chunk_size
    max_timesteps = int(TASK_MAX_STEPS[cfg.task_suite_name])

    if temporal_agg:
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + num_queries, action_dim],
            dtype=torch.float32,
        ).cuda()

    # ── Episode loop ──────────────────────────────────────────────────────
    t = 0
    replay_traj = dict()
    image_list = []
    robot_state_list = []
    target_action_list = []
    success = False

    with torch.inference_mode():
        try:
            # Stabilisation steps
            while t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
            t = 0

            while t < max_timesteps:
                traj_rgb_np, robot_state = get_obs(obs=obs, stats=stats)

                image_list.append(cv2.resize(traj_rgb_np[0], (256, 256)))
                robot_state_list.append(robot_state)
                robot_state = torch.from_numpy(robot_state).float().cuda()

                if t % query_frequency == 0:
                    curr_image = []
                    for img in traj_rgb_np:
                        curr_image.append(to_tensor(img).float().cuda())
                    curr_image = torch.stack(curr_image, dim=0)

                    if rand_crop_resize:
                        original_size = curr_image.shape[-2:]
                        ratio = 0.95
                        curr_image = curr_image[
                            :,
                            :,
                            int(original_size[0] * (1 - ratio) / 2) : int(
                                original_size[0] * (1 + ratio) / 2
                            ),
                            int(original_size[1] * (1 - ratio) / 2) : int(
                                original_size[1] * (1 + ratio) / 2
                            ),
                        ]
                        curr_image = curr_image.squeeze(0)
                        resize_transform = transforms.Resize(original_size, antialias=True)
                        curr_image = resize_transform(curr_image)
                        curr_image = curr_image.unsqueeze(0)

                # Network warm-up on first step
                if t == 0:
                    for _ in range(10):
                        batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                        policy.policy(**batch, eval=True)

                # Query policy
                if policy_config["action_head"] in ("act", "droid_diffusion"):
                    if t % query_frequency == 0:
                        batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                        all_actions = policy.policy(**batch, eval=True)

                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                else:
                    raise NotImplementedError

                # Post-process action
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action = convert_actions(action)

                # Execute in environment
                obs, reward, done, info = env.step(action.tolist())
                target_action_list.append(action)

                if done:
                    success = True
                    break
                t += 1

        except Exception as e:
            log_message(f"Episode error: {e}", log_file)
            success = False

    replay_traj["images"] = image_list
    replay_traj["task_command"] = task_description
    replay_traj["states"] = robot_state_list
    replay_traj["actions"] = target_action_list

    return success, replay_traj


# ── Single ablation variant runner ───────────────────────────────────────────

def run_ablation_task(
    cfg: GenerateConfig,
    task_key: str,
    task_info: dict,
    task_suite,
    task,
    policy,
    policy_config,
    log_file=None,
):
    """Run evaluation for a single ablation variant."""
    task_id = cfg.ablation_task_id

    log_message("=" * 80, log_file)
    log_message(f"ABLATION TASK: {task_key.upper()}", log_file)
    log_message(f"BDDL File: {task_info['bddl_file']}", log_file)
    log_message("=" * 80, log_file)

    # Load initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Init environment with ablation BDDL
    env, task_description, original_description = get_libero_env(
        task,
        cfg.model_family,
        ablation_bddl_file=task_info["bddl_file"],
        resolution=cfg.env_img_res,
    )

    ablation_command = task_description

    log_message(f"Original Task {task_id + 1} Command: {original_description}", log_file)
    log_message(f"Ablation Command (from BDDL): '{ablation_command}'", log_file)

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Ablation {task_key}"):

        # Select initial state
        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = original_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(
                    f"Skipping episode {episode_idx} (failed expert demo)", log_file
                )
                continue

            initial_state = np.array(
                all_initial_states[initial_states_task_key][episode_key]["initial_state"]
            )

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode with ablation command
        success, replay_traj = run_episode(
            cfg,
            env,
            ablation_command,
            policy,
            policy_config,
            initial_state,
            log_file,
        )

        task_episodes += 1
        if success:
            task_successes += 1

        # Save replay video
        save_rollout_video(
            replay_traj,
            task_episodes,
            success=success,
            task_description=f"ablation_task{task_id + 1}_{task_key}_{ablation_command.replace(' ', '_')}",
            log_file=log_file,
            dataset_name=cfg.task_suite_name,
            run=cfg.run_number,
            change_command=True,
            command_level="ablation",
        )

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {task_episodes}", log_file)
        log_message(
            f"# successes: {task_successes} ({task_successes / task_episodes * 100:.1f}%)",
            log_file,
        )

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
    log_message(f"Current task success rate: {task_success_rate:.4f}", log_file)

    if cfg.use_wandb:
        wandb.log({
            f"success_rate/ablation_{task_key}": task_success_rate,
            f"num_episodes/ablation_{task_key}": task_episodes,
        })

    return task_success_rate, task_episodes, task_successes


# ── Summary printer ───────────────────────────────────────────────────────────

def print_ablation_results(results, task_name: str, ablation_tests: dict):
    print("\n" + "=" * 100)
    print(f"ABLATION STUDY RESULTS - Task: {task_name}")
    print("=" * 100)
    print(f"{'Test':<20} | {'Command':<20} | {'Success Rate':>12} | {'Episodes':>15}")
    print("-" * 100)

    for task_key, result in results.items():
        cmd = ablation_tests[task_key]["expected_command"]
        sr = result["success_rate"]
        succ = result["successes"]
        total = result["episodes"]
        print(f"{task_key:<20} | {cmd:<20} | {sr:>11.1%} | {succ:>6}/{total:<7}")

    print("=" * 100)

    avg_sr = sum(r["success_rate"] for r in results.values()) / len(results)
    total_succ = sum(r["successes"] for r in results.values())
    total_eps = sum(r["episodes"] for r in results.values())
    print(f"{'AVERAGE':<20} | {'':<20} | {avg_sr:>11.1%} | {total_succ:>6}/{total_eps:<7}")
    print("=" * 100)


# ── Main entry point ──────────────────────────────────────────────────────────

@draccus.wrap()
def eval_ablation(cfg: GenerateConfig) -> float:
    """Main function for TinyVLA ablation study."""
    if cfg.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    # Build ablation config
    ablation_config = get_ablation_tasks(cfg.ablation_task_id)
    task_name = ablation_config["task_name"]
    ablation_tests = ablation_config["tests"]

    # ── Build TinyVLA policy ──────────────────────────────────────────────
    action_head = "droid_diffusion"
    policy_config = {
        "model_path": cfg.model_path,
        "model_base": cfg.model_base,
        "enable_lora": True,
        "conv_mode": "pythia",
        "action_head": action_head,
    }
    policy = llava_pythia_act_policy(policy_config)

    # ── Init LIBERO benchmark ─────────────────────────────────────────────
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.ablation_task_id)

    # ── Logging ───────────────────────────────────────────────────────────
    log_file, local_log_filepath, run_id = setup_logging(cfg, task_name)

    log_message("=" * 80, log_file)
    log_message(
        f"ABLATION STUDY: Task {cfg.ablation_task_id + 1} Keyword Shortcut Analysis",
        log_file,
    )
    log_message("=" * 80, log_file)
    log_message(f"Task Name:       {task_name}", log_file)
    log_message(f"Base Command:    {task.language}", log_file)
    log_message(f"Model:           {cfg.model_path}", log_file)
    log_message(f"Seed:            {cfg.seed}", log_file)
    log_message(f"Trials/ablation: {cfg.num_trials_per_task}", log_file)
    log_message(f"Ablation tests:  {list(ablation_tests.keys())}", log_file)
    log_message("=" * 80, log_file)

    # ── Filter to a single variant if ablation_test_key is provided ────────
    if cfg.ablation_test_key:
        if cfg.ablation_test_key not in ablation_tests:
            raise ValueError(
                f"Unknown ablation_test_key '{cfg.ablation_test_key}'. "
                f"Available keys for task {cfg.ablation_task_id}: {list(ablation_tests.keys())}"
            )
        ablation_tests = {cfg.ablation_test_key: ablation_tests[cfg.ablation_test_key]}
        log_message(f"Running single variant: {cfg.ablation_test_key}", log_file)
    else:
        log_message(f"Running all variants: {list(ablation_tests.keys())}", log_file)

    # ── Run all ablation variants ─────────────────────────────────────────
    results = {}
    for task_key, task_info in ablation_tests.items():
        sr, episodes, successes = run_ablation_task(
            cfg,
            task_key,
            task_info,
            task_suite,
            task,
            policy,
            policy_config,
            log_file,
        )
        results[task_key] = {
            "success_rate": sr,
            "episodes": episodes,
            "successes": successes,
        }

    # ── Summary ───────────────────────────────────────────────────────────
    print_ablation_results(results, task_name, ablation_tests)

    log_message("\n" + "=" * 80, log_file)
    log_message("FINAL RESULTS", log_file)
    log_message("=" * 80, log_file)
    for task_key, result in results.items():
        log_message(
            f"  {task_key}: {result['success_rate']:.1%} "
            f"({result['successes']}/{result['episodes']})",
            log_file,
        )
    avg_sr = sum(r["success_rate"] for r in results.values()) / len(results)
    log_message(f"\nAVERAGE: {avg_sr:.1%}", log_file)
    log_message("=" * 80, log_file)

    if cfg.use_wandb:
        wandb.log({"success_rate/ablation_average": avg_sr})
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return avg_sr


if __name__ == "__main__":
    eval_ablation()
