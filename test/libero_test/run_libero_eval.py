import sys
import os
import logging
import glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
from llava_pythia.conversation import conv_templates, SeparatorStyle
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
import torch
from torchvision import transforms
import cv2
from copy import deepcopy
from itertools import repeat
from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import numpy as np
import time
from llava_pythia.model import *
from einops import rearrange
import torch_utils as TorchUtils
import matplotlib.pyplot as plt
import argparse
from robot_utils import set_seed_everywhere
from enum import Enum
from dataclasses import dataclass
import tqdm
from typing import Optional, Union
from dataclasses import dataclass
import draccus
import transformers
from libero.libero import benchmark
from libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from PIL import Image
from robot_utils import DATE_TIME
import wandb
import json
from collections import deque
import pickle


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)



# Libero Tasks
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"



# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}



def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
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
    pred_action = np.concatenate((cur_xyz, cur_euler, cur_gripper))


    return pred_action



def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    normalized_action = action.copy()
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1


    if binarize:
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])


    return normalized_action



def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    inverted_action = action.copy()
    inverted_action[..., -1] *= -1.0
    return inverted_action



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
        """Processes a batch of data for Llava-Pythia model input."""
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
            expanded_imgs = pil_imgs.permute(0,2,3,1).cpu().numpy()
        elif height > width:
            offset = (max_dim - width) // 2
            expanded_imgs[:, :height, offset:offset + width, :] = pil_imgs.permute(0,2,3,1).cpu().numpy()
        else:
            offset = (max_dim - height) // 2
            expanded_imgs[:, offset:offset + height, :width, :] = pil_imgs.permute(0,2,3,1).cpu().numpy()
       
        expanded_imgs = torch.tensor(expanded_imgs).to(dtype=pil_imgs.dtype, device=pil_imgs.device)
        return expanded_imgs



@dataclass
class GenerateConfig:
    # fmt: off


    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_path: str = "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tinyvla/tiny_vla_llava_pythia_lora_libero_goal_no_noops_lora_r_64/checkpoint-54000"                   # Path to the model
    model_base: str = "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tinyvla/llava_pythia_libero_goal_no_noops_64/1.3B"                    #
    model_family: str = "tiny_vla"  
   
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_GOAL     # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)


    task_range: str = "0-9"                          # Range of tasks to evaluate (e.g., "0-9" for tasks 0 to 9)
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    summary_file: Optional[str] = None               # Path to summary CSV file


    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project


    seed: int = 7                                    # Random Seed (for reproducibility)


    run_number: int = 0                                  # Run number (for logging purposes)
    debug: bool = False
    # fmt: on
    local_rank: int = 0  # Local rank for distributed training (default is 0)
    ##################################################################################################################
    # Command modification parameters
    ##################################################################################################################
    change_command: bool = False                     # Whether to change the command during evaluation
    command_level: Optional[str] = None              # Command level: 'l1', 'l2', 'l3', 'all', 'all_no_default', 'default', or None


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
    # normalize states
    states = (states - stats["qpos_mean"]) / stats["qpos_std"]
    return images, states


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
   
    # Add command level to run_id if specified
    if cfg.change_command and cfg.command_level:
        run_id += f"--{cfg.command_level}"


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



def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    policy,
    policy_config,
    resize_size,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    env.reset()
    to_tensor = transforms.ToTensor()
   
    # Set initial state
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.reset()
        # Fix basket position if needed (TinyVLA-specific)
        if 'basket_1_pos' in obs.keys():
            basket_pos = [0.005, 0.261, 0.035]
            basket_quat = [0.000, 0.000, 0.000, 1.000]
            env.sim.data.set_joint_qpos(
                env.env.objects_dict['basket_1'].joints[0],
                np.concatenate((basket_pos, basket_quat))
            )
            t = 0
            while t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
   
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
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
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
               
                image_list.append(cv2.resize(traj_rgb_np[0], (256,256)))
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
                    # Warm up
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


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id,
    policy,
    policy_config,
    log_file,
    total_episodes=0,
    total_successes=0
):
    """Run evaluation for a single task."""
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)


    # Initialize environment (returns 3 values with L1/L2/L3 support)
    env, task_description, original_description = get_libero_env(
        task,
        cfg.model_family,
        change_command=cfg.change_command,
        command_level=cfg.command_level,
        resolution=cfg.env_img_res
    )


    # Log task info
    log_message("=" * 80, log_file)
    log_message(f"TASK {task_id + 1}/{task_suite.n_tasks}", log_file)
    log_message(f"Original Command: {original_description}", log_file)
   
    if cfg.change_command and cfg.command_level:
        log_message(f"Command Level: {cfg.command_level.upper()}", log_file)
        log_message(f"Variation Command: {task_description}", log_file)
        if task_description == original_description:
            log_message("WARNING: Variation same as original - check BDDL file", log_file)
    else:
        log_message(f"Command Level: DEFAULT", log_file)
   
    log_message("=" * 80, log_file)


    # ✅ SEMPRE 50 EPISODI NUOVI - NO SKIP
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)
       
        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"


            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping episode {episode_idx} (failed expert demo)", log_file)
                continue


            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])


        log_message(f"Starting episode {task_episodes + 1}...", log_file)


        # Run episode
        success, replay_traj = run_episode(
            cfg, env, task_description, policy, policy_config, 224, initial_state, log_file
        )
       
        # ✅ CONTATORI SEMPRE CORRETTI
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1


        save_rollout_video(
            replay_traj,
            total_episodes,
            success=success,
            task_description=task_description,
            log_file=log_file,
            dataset_name=cfg.task_suite_name,
            run=cfg.run_number,
            change_command=cfg.change_command,
            command_level=cfg.command_level
        )
       
        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)


    # Task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    log_message(f"Task success rate: {task_success_rate:.4f}", log_file)


    if cfg.use_wandb:
        wandb.log({
            f"success_rate/{task_description}": task_success_rate,
            f"num_episodes/{task_description}": task_episodes,
        })


    return total_episodes, total_successes, task_description, task_success_rate, task_episodes


def print_results_table(task_results, command_levels, all_results):
    """Print summary table of results with success counts."""
    print("\n" + "=" * 100)
    print("DETAILED RESULTS TABLE")
    print("=" * 100)
   
    first_level = command_levels[0]
    level_name = first_level if first_level is not None else "default"
    task_names = list(task_results[level_name].keys())
    level_names = [l if l is not None else "default" for l in command_levels]
   
    if len(level_names) == 1:
        # Single level: show task name, success rate, and counts
        print(f"{'Task':<50} | {'Success Rate':>20} | {'Episodes':>8}")
        print("-" * 100)
       
        for task_name in task_names:
            result = task_results[level_names[0]][task_name]
            sr = result['success_rate']
            eps = result['episodes']
            successes = int(sr * eps)  # Calculate number of successes
            print(f"{task_name:<50} | {sr:>11.1%} ({successes:>2}/{eps:<2}) | {eps:>8}")
       
        print("-" * 100)
        overall_sr = all_results[level_names[0]]['success_rate']
        overall_succ = all_results[level_names[0]]['total_successes']
        overall_eps = all_results[level_names[0]]['total_episodes']
        print(f"{'OVERALL':<50} | {overall_sr:>11.1%} ({overall_succ:>3}/{overall_eps:<3}) | {overall_eps:>8}")
   
    else:
        # Multiple levels: show task name and success rate with counts for each level
        header = f"{'Task':<40}"
        for level_name in level_names:
            header += f" | {level_name.upper():>20}"
        print(header)
        print("-" * (41 + len(level_names) * 24))
       
        for task_name in task_names:
            row = f"{task_name:<40}"
            for level_name in level_names:
                if task_name in task_results[level_name]:
                    result = task_results[level_name][task_name]
                    sr = result['success_rate']
                    eps = result['episodes']
                    successes = int(sr * eps)  # Calculate number of successes
                    row += f" | {sr:>6.1%} ({successes:>2}/{eps:<2})"
                else:
                    row += f" | {'N/A':>20}"
            print(row)
       
        print("-" * (41 + len(level_names) * 24))
       
        # Overall row
        overall_row = f"{'OVERALL':<40}"
        for level_name in level_names:
            result = all_results[level_name]
            sr = result['success_rate']
            succ = result['total_successes']
            total = result['total_episodes']
            overall_row += f" | {sr:>6.1%} ({succ:>3}/{total:<3})"
        print(overall_row)
   
    print("=" * 100)
   
    # Summary statistics
    print("\nSUMMARY BY COMMAND LEVEL:")
    print("-" * 60)
    for level_name in level_names:
        result = all_results[level_name]
        sr = result['success_rate']
        succ = result['total_successes']
        total = result['total_episodes']
        print(f"  {level_name.upper():>15}: {sr:.1%} ({succ}/{total} episodes)")
    print("=" * 100)



@draccus.wrap()
def run_libero_eval(cfg: GenerateConfig):
    """Main evaluation function with L1/L2/L3 support."""
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()


    # TinyVLA policy setup
    action_head = 'droid_diffusion'
    policy_config = {
        "model_path": f"{cfg.model_path}",
        "model_base": f"{cfg.model_base}",
        "enable_lora": True,
        "conv_mode": "pythia",
        "action_head": action_head,
    }
   
    set_seed_everywhere(cfg.seed)
    policy = llava_pythia_act_policy(policy_config)


    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    print(f"Evaluating {num_tasks} tasks in {cfg.task_suite_name}")
   
    task_start, task_end = map(int, cfg.task_range.split('-'))
    print(f"Task range: {task_start}-{task_end}")


    # Determine command levels to test
    if cfg.change_command and cfg.command_level == "all":
        command_levels = [None, "l1", "l2", "l3"]
    elif cfg.change_command and cfg.command_level == "all_no_default":
        command_levels = ["l1", "l2", "l3"]
    elif cfg.change_command and cfg.command_level == "default":
        command_levels = [None]
    elif cfg.change_command and cfg.command_level is not None:
        command_levels = [cfg.command_level]
    else:
        command_levels = [None]


    all_results = {}
    task_results = {}


    # Loop over command levels
    for level in command_levels:
        current_level_name = level if level is not None else "default"
        cfg.command_level = level
        cfg.change_command = (level is not None)
        
        # Reset seed for each level to ensure reproducibility
        set_seed_everywhere(cfg.seed)
       
        log_file, local_log_filepath, run_id = setup_logging(cfg)
       
        log_message("=" * 80, log_file)
        log_message(f"EVALUATING: {current_level_name.upper()}", log_file)
        log_message("=" * 80, log_file)


        task_results[current_level_name] = {}


        total_episodes, total_successes = 0, 0
        for task_id in tqdm.tqdm(range(task_start, min(task_end+1, num_tasks)), desc=f"Level {current_level_name}"):
            total_episodes, total_successes, task_name, task_sr, task_eps = run_task(
                cfg, task_suite, task_id, policy, policy_config,
                log_file, total_episodes, total_successes
            )
           
            task_results[current_level_name][task_name] = {
                'success_rate': task_sr,
                'episodes': task_eps
            }


        final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
        all_results[current_level_name] = {
            'success_rate': final_success_rate,
            'total_episodes': total_episodes,
            'total_successes': total_successes
        }


        log_message("=" * 80, log_file)
        log_message(f"RESULTS FOR {current_level_name.upper()}:", log_file)
        log_message(f"Success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
        log_message("=" * 80, log_file)
       
        if cfg.summary_file:
            summary_data = {
                'task_range': f"{task_start}-{task_end}",
                'task_results': task_results[current_level_name],
                'overall_results': all_results[current_level_name]
            }
            with open(cfg.summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            print(f"PARTIAL RESULTS SAVED: {cfg.summary_file}")


        if cfg.use_wandb:
            wandb.log({
                f"success_rate/{current_level_name}": final_success_rate,
                f"num_episodes/{current_level_name}": total_episodes,
            })
            wandb.save(local_log_filepath)


        if log_file:
            log_file.close()


    print_results_table(task_results, command_levels, all_results)


    if len(command_levels) > 1:
        return sum(r['success_rate'] for r in all_results.values()) / len(all_results)
    else:
        return final_success_rate



if __name__ == '__main__':
    run_libero_eval()