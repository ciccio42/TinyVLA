"""
run_libero_eval_task_comp.py

Evaluates a TinyVLA trained policy on custom LIBERO task composition scenarios (task_comp_l1).
Tests task-level generalization: the model must apply known primitives to new object/target
combinations never seen during training.

Custom tasks (all share the libero_goal scene):
  1. Put the plate on the top of the cabinet
  2. Put the plate on the stove
  3. Put the cream cheese on the top of the cabinet
  4. Put the cream cheese on the plate
  5. Open the top layer of the drawer and put the cream cheese inside
"""

import sys
import os
import logging
import gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"

from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
from llava_pythia.conversation import conv_templates, SeparatorStyle
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

import torch
from torchvision import transforms
import cv2
from copy import deepcopy
import numpy as np
import time
from llava_pythia.model import *
from einops import rearrange
import torch_utils as TorchUtils
import pickle

import draccus
from dataclasses import dataclass
from typing import Optional, Union
from enum import Enum
import tqdm
import json
from collections import deque

from libero.libero import get_libero_path
from libero.libero.benchmark import Task
from libero.libero.envs import OffScreenRenderEnv

from libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    extract_command_from_bddl,
)
from robot_utils import DATE_TIME, set_seed_everywhere


# ============================================================================
# Task Composition L1 - Custom Task Definitions
# ============================================================================

TASK_COMP_L1_TASKS = [
    {
        # Training: push plate (push_plate_to_stove) + put on cabinet (bowl/wine_bottle→cabinet)
        # Composition: pick-place plate → cabinet (new object-target pair)
        "bddl_file": "put_the_plate_on_top_of_the_cabinet_task_comp_l1.bddl",
        "init_states_from": "push_the_plate_to_the_front_of_the_stove",
    },
    {
        # Training: push plate (push_plate_to_stove) + put on stove (bowl→stove)
        # Composition: pick-place plate → stove (new object-target pair)
        "bddl_file": "put_the_plate_on_the_stove_task_comp_l1.bddl",
        "init_states_from": "push_the_plate_to_the_front_of_the_stove",
    },
    {
        # Training: cream_cheese→bowl + put on cabinet (bowl/wine_bottle→cabinet)
        # Composition: pick-place cream_cheese → cabinet (new object-target pair)
        "bddl_file": "put_the_cream_cheese_on_top_of_the_cabinet_task_comp_l1.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
    {
        # Training: cream_cheese→bowl + bowl→plate
        # Composition: pick-place cream_cheese → plate (new object-target pair)
        "bddl_file": "put_the_cream_cheese_on_the_plate_task_comp_l1.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
    {
        # Training: open drawer + bowl inside
        # Composition: open drawer + cream_cheese inside (swaps object, same primitive)
        "bddl_file": "open_the_top_drawer_and_put_the_cream_cheese_inside_task_comp_l1.bddl",
        "init_states_from": "open_the_top_drawer_and_put_the_bowl_inside",
    },
]

TASK_MAX_STEPS = 300  # same as libero_goal


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def convert_actions(pred_action):
    cur_xyz = pred_action[:3]
    cur_rot6d = pred_action[3:9]
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)
    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)
    cur_euler = TorchUtils.rot_6d_to_euler_angles(rot_6d=cur_rot6d, convention="XYZ").squeeze().numpy()
    return np.concatenate((cur_xyz, cur_euler, cur_gripper))


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    normalized_action = action.copy()
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])
    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    inverted_action = action.copy()
    inverted_action[..., -1] *= -1.0
    return inverted_action


def get_obs(obs, stats):
    """Retrieves observations from the robot environment."""
    images = np.array([
        cv2.resize(obs['agentview_image'][::-1, ::-1], (320, 180)),
        cv2.resize(obs['robot0_eye_in_hand_image'][::-1, ::-1], (320, 180))
    ])
    states = np.concatenate((
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"]
    ))
    states = (states - stats["qpos_mean"]) / stats["qpos_std"]
    return images, states


# ============================================================================
# Policy Class (from TinyVLA)
# ============================================================================

class llava_pythia_act_policy:
    """Policy class for Llava-Pythia action generation."""

    def __init__(self, policy_config, data_args=None):
        super(llava_pythia_act_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        model_base = policy_config["model_base"] if policy_config['enable_lora'] else None
        model_name = get_model_name_from_path(policy_config['model_path'])
        model_path = policy_config["model_path"]

        self.tokenizer, self.policy, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name, False, False
        )
        self.config = LlavaPythiaConfig.from_pretrained(
            '/'.join(model_path.split('/')[:-1]), trust_remote_code=True
        )

    def process_batch_to_llava(self, curr_image, robo_state, raw_lang):
        self.conv = conv_templates[self.policy_config['conv_mode']].copy()

        if len(curr_image.shape) == 5:
            curr_image = curr_image.squeeze(0)

        image, image_r = torch.chunk(curr_image, 2, dim=0)

        image = self.expand2square(image, tuple(x for x in self.image_processor.image_mean))
        image_tensor = self.image_processor.preprocess(
            image, return_tensors='pt', do_normalize=True, do_rescale=False, do_center_crop=False
        )['pixel_values']
        image_tensor = image_tensor.to(self.policy.device, dtype=self.policy.dtype)

        image_r = self.expand2square(image_r, tuple(x for x in self.image_processor.image_mean))
        image_tensor_r = self.image_processor.preprocess(
            image_r, return_tensors='pt', do_normalize=True, do_rescale=False, do_center_crop=False
        )['pixel_values']
        image_tensor_r = image_tensor_r.to(self.policy.device, dtype=self.policy.dtype)

        inp = raw_lang
        assert image is not None, 'image must be provided.'

        if self.policy.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        self.conv.append_message(self.conv.roles[0], inp)
        image = None

        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        prompt += " <|endoftext|>"

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        states = robo_state.to(self.policy.device, dtype=self.policy.dtype)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attn_mask,
            images=image_tensor,
            images_r=image_tensor_r,
            states=states.unsqueeze(0)
        )
        return data_dict

    def expand2square(self, pil_imgs, background_color):
        batch_size, channels, height, width = pil_imgs.shape
        max_dim = max(height, width)
        expanded_imgs = np.full((batch_size, max_dim, max_dim, channels), background_color, dtype=np.float32)

        if height == width:
            expanded_imgs = pil_imgs.permute(0, 2, 3, 1).cpu().numpy()
        elif height > width:
            offset = (max_dim - width) // 2
            expanded_imgs[:, :height, offset:offset + width, :] = pil_imgs.permute(0, 2, 3, 1).cpu().numpy()
        else:
            offset = (max_dim - height) // 2
            expanded_imgs[:, offset:offset + height, :width, :] = pil_imgs.permute(0, 2, 3, 1).cpu().numpy()

        expanded_imgs = torch.tensor(expanded_imgs).to(dtype=pil_imgs.dtype, device=pil_imgs.device)
        return expanded_imgs


# ============================================================================
# Configuration
# ============================================================================

class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


@dataclass
class GenerateConfig:
    # fmt: off

    # Model
    model_path: str = ""
    model_base: str = ""
    model_family: str = "tiny_vla"

    # LIBERO environment
    task_suite_name: str = TaskSuite.LIBERO_GOAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    env_img_res: int = 256

    # Utils
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    summary_file: Optional[str] = None
    checkpoint_size: int = 20000

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    seed: int = 7
    run_number: int = 0
    debug: bool = False
    local_rank: int = 0
    # fmt: on


# ============================================================================
# Custom Task Loading
# ============================================================================

def load_custom_tasks():
    """Build Task NamedTuples and load init_states for each task_comp_l1 task."""
    bddl_dir = os.path.join(get_libero_path("bddl_files"), "libero_goal")
    init_dir = os.path.join(get_libero_path("init_states"), "libero_goal")

    custom_tasks = []
    for task_def in TASK_COMP_L1_TASKS:
        bddl_filename = task_def["bddl_file"]
        init_from = task_def["init_states_from"]

        bddl_path = os.path.join(bddl_dir, bddl_filename)
        assert os.path.exists(bddl_path), f"BDDL file not found: {bddl_path}"

        task_description = extract_command_from_bddl(bddl_path)
        assert task_description is not None, f"Could not extract language from {bddl_path}"

        task_name = bddl_filename.replace(".bddl", "")
        task = Task(
            name=task_name,
            language=task_description,
            problem="Libero",
            problem_folder="libero_goal",
            bddl_file=bddl_filename,
            init_states_file=f"{init_from}.pruned_init",
        )

        init_states_path = os.path.join(init_dir, f"{init_from}.pruned_init")
        assert os.path.exists(init_states_path), f"Init states not found: {init_states_path}"
        init_states = torch.load(init_states_path, weights_only=False)

        custom_tasks.append({
            "task": task,
            "init_states": init_states,
            "task_description": task_description,
            "bddl_path": bddl_path,
        })

    return custom_tasks


def create_env_from_bddl(bddl_path, resolution=256):
    """Create LIBERO environment directly from a BDDL file path."""
    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-task_comp_l1-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    if cfg.checkpoint_size > 0:
        run_id += f"--ckpt{cfg.checkpoint_size}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        import wandb
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    return log_file, local_log_filepath, run_id


# ============================================================================
# Episode & Task Execution
# ============================================================================

def run_episode(
    cfg, env, task_description, policy, policy_config, resize_size,
    initial_state=None, log_file=None,
):
    """Run a single episode in the environment."""
    env.reset()
    to_tensor = transforms.ToTensor()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.reset()

    # Policy configuration
    if policy_config["action_head"] == 'act':
        rand_crop_resize = False
        temporal_agg = True
    else:
        rand_crop_resize = True
        temporal_agg = True

    action_dim = policy.config.action_dim
    policy.policy.eval()

    # Load stats
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # Post-processing
    if policy_config["action_head"] == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif policy_config["action_head"] == 'transformer_diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    elif policy_config["action_head"] == 'droid_diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']

    query_frequency = policy.config.chunk_size / 2
    if temporal_agg:
        query_frequency = 1
        num_queries = policy.config.chunk_size
    max_timesteps = int(200)

    if temporal_agg:
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + num_queries, action_dim],
            dtype=torch.float32
        ).cuda()

    t = 0
    replay_traj = dict()
    image_list = []
    robot_state_list = []
    target_action_list = []
    success = False

    with torch.inference_mode():
        try:
            # Wait for stabilization
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
                        curr_image = curr_image[:, :,
                                        int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                                        int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                        curr_image = curr_image.squeeze(0)
                        resize_transform = transforms.Resize(original_size, antialias=True)
                        curr_image = resize_transform(curr_image)
                        curr_image = curr_image.unsqueeze(0)

                if t == 0:
                    for _ in range(10):
                        batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                        policy.policy(**batch, eval=True)

                # Query policy
                if policy_config['action_head'] in ["act", "droid_diffusion"]:
                    if t % query_frequency == 0:
                        batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                        all_actions = policy.policy(**batch, eval=True)

                    if temporal_agg:
                        all_time_actions[[t], t:t + num_queries] = all_actions
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

                # Post-process
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action = convert_actions(action)

                # Execute
                obs, reward, done, info = env.step(action.tolist())
                target_action_list.append(action)

                if done:
                    success = True
                    break
                t += 1

        except Exception as e:
            log_message(f"Episode error: {e}", log_file)
            success = False

    replay_traj['images'] = image_list
    replay_traj['task_command'] = task_description
    replay_traj['states'] = robot_state_list
    replay_traj['actions'] = target_action_list

    return success, replay_traj


def run_custom_task(
    cfg, task_info, task_idx, num_tasks, policy, policy_config,
    log_file, total_episodes=0, total_successes=0,
):
    """Run evaluation for a single custom task_comp_l1 task."""
    task = task_info["task"]
    init_states = task_info["init_states"]
    task_description = task_info["task_description"]
    bddl_path = task_info["bddl_path"]

    # Create environment from custom BDDL
    env = create_env_from_bddl(bddl_path, resolution=cfg.env_img_res)

    log_message("=" * 80, log_file)
    log_message(f"TASK {task_idx + 1}/{num_tasks} (Task Composition L1)", log_file)
    log_message(f"BDDL: {task.bddl_file}", log_file)
    log_message(f"Command: {task_description}", log_file)
    log_message(f"Init states from: {task.init_states_file}", log_file)
    log_message("=" * 80, log_file)

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        initial_state = init_states[episode_idx]

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        success, replay_traj = run_episode(
            cfg, env, task_description, policy, policy_config,
            224, initial_state, log_file,
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save rollout video - use custom directory for task_comp_l1
        rollout_dir = f"/mnt/beegfs/a.cardamone7/outputs/rollouts/libero_goal/task_composition/tinyvla/task_comp_l1/run_{cfg.run_number}"
        os.makedirs(rollout_dir, exist_ok=True)
        processed_desc = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
        mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={total_episodes}--success={success}--task={processed_desc}.mp4"

        import imageio
        video_writer = imageio.get_writer(mp4_path, fps=30)
        for img in replay_traj['images']:
            video_writer.append_data(img)
        video_writer.close()
        log_message(f"Saved rollout MP4 at {mp4_path}", log_file)

        npy_path = mp4_path.replace('.mp4', '.npy')
        np.save(npy_path, replay_traj)
        log_message(f"Saved trajectory at {npy_path}", log_file)

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    log_message(f"Task success rate: {task_success_rate:.4f} ({task_success_rate * 100:.1f}%)", log_file)

    if cfg.use_wandb:
        import wandb
        wandb.log({
            f"success_rate/{task_description}": task_success_rate,
            f"num_episodes/{task_description}": task_episodes,
        })

    # Cleanup
    try:
        env.close()
        log_message("Environment closed successfully", log_file)
    except Exception as e:
        log_message(f"Warning: Error closing environment: {e}", log_file)
    gc.collect()

    return total_episodes, total_successes, task_description, task_success_rate, task_episodes


# ============================================================================
# Results
# ============================================================================

def print_results_table(task_results, all_results):
    """Print a summary table of task composition L1 results."""
    print("\n" + "=" * 100)
    print("TASK COMPOSITION L1 (TinyVLA) - RESULTS TABLE")
    print("=" * 100)

    print(f"{'Task':<60} | {'Success Rate':>20} | {'Episodes':>8}")
    print("-" * 100)

    for task_name, result in task_results.items():
        sr = result['success_rate']
        eps = result['episodes']
        successes = int(sr * eps)
        print(f"{task_name:<60} | {sr:>11.1%} ({successes:>2}/{eps:<2}) | {eps:>8}")

    print("-" * 100)
    overall_sr = all_results['success_rate']
    overall_succ = all_results['total_successes']
    overall_eps = all_results['total_episodes']
    print(f"{'OVERALL':<60} | {overall_sr:>11.1%} ({overall_succ:>3}/{overall_eps:<3}) | {overall_eps:>8}")
    print("=" * 100)


# ============================================================================
# Main Entry Point
# ============================================================================

@draccus.wrap()
def eval_task_comp(cfg: GenerateConfig) -> float:
    """Evaluate TinyVLA on task composition L1 scenarios."""
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    assert cfg.model_path, "model_path must not be empty!"
    assert cfg.model_base, "model_base must not be empty!"

    # Set seed
    set_seed_everywhere(cfg.seed)

    # Initialize policy
    action_head = 'droid_diffusion'
    policy_config = {
        "model_path": cfg.model_path,
        "model_base": cfg.model_base,
        "enable_lora": True,
        "conv_mode": "pythia",
        "action_head": action_head,
    }
    policy = llava_pythia_act_policy(policy_config)

    # Load custom tasks
    custom_tasks = load_custom_tasks()
    num_tasks = len(custom_tasks)

    log_message(f"Loaded {num_tasks} task composition L1 tasks", None)
    for i, ct in enumerate(custom_tasks):
        log_message(f"  [{i}] {ct['task_description']} ({ct['task'].bddl_file})", None)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    log_message("=" * 80, log_file)
    log_message("TASK COMPOSITION L1 EVALUATION (TinyVLA)", log_file)
    log_message(f"Model: {cfg.model_path}", log_file)
    log_message(f"Model base: {cfg.model_base}", log_file)
    log_message(f"Seed: {cfg.seed}", log_file)
    log_message(f"Num trials per task: {cfg.num_trials_per_task}", log_file)
    log_message(f"Num tasks: {num_tasks}", log_file)
    log_message("=" * 80, log_file)

    # Run evaluation
    total_episodes, total_successes = 0, 0
    task_results = {}

    for task_idx in tqdm.tqdm(range(num_tasks), desc="Task Comp L1"):
        total_episodes, total_successes, task_name, task_sr, task_eps = run_custom_task(
            cfg, custom_tasks[task_idx], task_idx, num_tasks,
            policy, policy_config, log_file,
            total_episodes, total_successes,
        )

        task_results[task_name] = {
            'success_rate': task_sr,
            'episodes': task_eps,
        }

    # Final results
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    all_results = {
        'success_rate': final_success_rate,
        'total_episodes': total_episodes,
        'total_successes': total_successes,
    }

    log_message("=" * 80, log_file)
    log_message("FINAL RESULTS - TASK COMPOSITION L1 (TinyVLA):", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    log_message("=" * 80, log_file)

    # Save summary
    if cfg.summary_file:
        summary_data = {
            'task_results': task_results,
            'overall_results': all_results,
        }
        os.makedirs(os.path.dirname(cfg.summary_file), exist_ok=True)
        with open(cfg.summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Summary saved to: {cfg.summary_file}")

    if cfg.use_wandb:
        import wandb
        wandb.log({
            "success_rate/task_comp_l1_overall": final_success_rate,
            "num_episodes/task_comp_l1_overall": total_episodes,
        })
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    print_results_table(task_results, all_results)

    return final_success_rate


if __name__ == "__main__":
    eval_task_comp()
