import sys
import os
import logging
import glob
sys.path.append("../..")
sys.path.append("/home/rsofnc000/Multi-Task-LFD-Framework/repo/LIBERO")
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
from utils.utils import set_seed_everywhere
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
# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}

def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def convert_actions(pred_action):
    # pred_action = torch.from_numpy(actions)
    # pred_action = actions.squeeze(0)
    cur_xyz = pred_action[:3]
    cur_rot6d = pred_action[3:9]
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)

    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)
    cur_euler = TorchUtils.rot_6d_to_euler_angles(rot_6d=cur_rot6d, convention="XYZ").squeeze().numpy()
    # print(f'cur_xyz size: {cur_xyz.shape}')
    # print(f'cur_euler size: {cur_euler.shape}')
    # print(f'cur_gripper size: {cur_gripper.shape}')
    pred_action = np.concatenate((cur_xyz, cur_euler, cur_gripper))
    # print(f'4. pred_action size: {pred_action.shape}')
    # print(f'4. after convert pred_action: {pred_action}')

    return pred_action


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action

def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action


class llava_pythia_act_policy:
    """
    Policy class for Llava-Pythia action generation.

    Attributes:
        policy_config: Configuration dictionary for the policy.
    """
    def __init__(self, policy_config, data_args=None):
        super(llava_pythia_act_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        # self.conv = conv_templates[policy_config['conv_mode']].copy()
        model_base = policy_config["model_base"] if policy_config[
            'enable_lora'] else None
        model_name = get_model_name_from_path(policy_config['model_path'])
        model_path = policy_config["model_path"]

        self.tokenizer, self.policy, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, model_name, False, False)
        self.config = LlavaPythiaConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=True)

    def process_batch_to_llava(self, curr_image, robo_state, raw_lang):
        """
        Processes a batch of data for Llava-Pythia model input.

        Args:
            curr_image: Current image tensor.
            robo_state: Current robot state tensor.
            raw_lang: Raw language input.

        Returns:
            A dictionary containing processed data for the model.
        """
        self.conv = conv_templates[self.policy_config['conv_mode']].copy()

        if len(curr_image.shape) == 5: # 1,2,3,270,480
            curr_image = curr_image.squeeze(0)

        # for k,v in sample.items():
        #     print(k, v.shape)
        image, image_r = torch.chunk(curr_image, 2, dim=0)

        image = self.expand2square(image, tuple(x for x in self.image_processor.image_mean))
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt', do_normalize=True, do_rescale=False, do_center_crop=False)['pixel_values']
        image_tensor = image_tensor.to(self.policy.device, dtype=self.policy.dtype)

        image_r = self.expand2square(image_r, tuple(x for x in self.image_processor.image_mean))
        image_tensor_r = self.image_processor.preprocess(image_r, return_tensors='pt', do_normalize=True, do_rescale=False, do_center_crop=False)['pixel_values']
        image_tensor_r = image_tensor_r.to(self.policy.device, dtype=self.policy.dtype)

        # print('raw_lang')
        inp = raw_lang
        assert image is not None, 'image must be provided.'
        # first message
        if self.policy.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        self.conv.append_message(self.conv.roles[0], inp)
        image = None

        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        prompt += " <|endoftext|>"

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        states = robo_state.to(self.policy.device, dtype=self.policy.dtype)
        # print(input_ids.dtype, attn_mask.dtype, image_tensor.dtype, image_tensor_r.dtype, states.dtype)

        data_dict = dict(input_ids=input_ids,
                         attention_mask=attn_mask,
                         images=image_tensor,
                         images_r=image_tensor_r,
                         states=states.unsqueeze(0))  # Add batch dimension

        # print(f"@@@@@@@@@@@@@@@{image_tensor.shape}")
        return data_dict

    def expand2square(self, pil_imgs, background_color):
        batch_size, channels, height, width = pil_imgs.shape
        max_dim = max(height, width)
        expanded_imgs = np.full((batch_size, max_dim, max_dim, channels), background_color, dtype=np.float32)

        if height == width:
            expanded_imgs = pil_imgs.permute(0,2,3,1).cpu().numpy()
        elif height > width:
            offset = (max_dim - width) // 2
            # expanded_imgs[:, :height, offset:offset + width] = pil_imgs
            expanded_imgs[:, :height, offset:offset + width, :] = pil_imgs.permute(0,2,3,1).cpu().numpy()
        else:
            offset = (max_dim - height) // 2
            # expanded_imgs[:, offset:offset + height, :width] = pil_imgs
            expanded_imgs[:, offset:offset + height, :width, :] = pil_imgs.permute(0,2,3,1).cpu().numpy()
        expanded_imgs = torch.tensor(expanded_imgs).to(dtype=pil_imgs.dtype, device=pil_imgs.device) # B H W C
        return expanded_imgs


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_path: str = "/home/rsofnc000/checkpoint_save_folder/tiny_vla/post-processed/checkpoint-3000"                   # Path to the model
    model_base: str = "/home/rsofnc000/checkpoint_save_folder/tiny_vla/post-processed/Llava-Pythia-1.3B"                    #
    model_family: str = "tiny_vla"  
    
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_OBJECT  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    run_number: int = 0                                  # Run number (for logging purposes)
    debug: bool = False  
    change_spawn: bool = False  # Whether to change spawn region of target object in the environment
    spawn_train_distribution: bool = False  # Whether to use the training spawn distribution for the target object
    # fmt: on
    local_rank: int = 0  # Local rank for distributed training (default is 0)


def get_obs(obs, stats):
    """
    Retrieves observations (images and robot states) from the robot environment.

    Returns:
        A tuple containing images and states.
    """

    images = np.array([cv2.resize(obs['agentview_image'][::-1, ::-1], (320, 180)),
                       cv2.resize(obs['robot0_eye_in_hand_image'][::-1, ::-1], (320, 180))])  # front image, wrist image
    states = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
    # normalize states
    states = (states - stats["qpos_mean"]) / stats["qpos_std"]
    return images, states

 
def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None, total_episodes=0):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
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
    # Reset environment
    env.reset()
    to_tensor = transforms.ToTensor()
    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        # env.set_init_state(initial_state)
        obs =  env.reset() #env.get_observation()
        # fix the position of the bin
        # due to problem with the initialization of the environment
        # For test with different spawn regions this is not a problem
        if 'basket_1_pos' in obs.keys():
            basket_pos = [0.005, 0.261, 0.035]
            basket_quat = [0.000, 0.000, 0.000, 1.000]  # [x, y, z, w]
            # env.sim.data.set_joint_qpos()
            env.sim.data.set_joint_qpos(env.env.objects_dict['basket_1'].joints[0], 
                                        np.concatenate((basket_pos, basket_quat)))
            t = 0
            while t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
    
    if policy_config["action_head"] == 'act':
        rand_crop_resize = False
        temporal_agg = True
    else:
        rand_crop_resize = True
        temporal_agg = True        
    
    action_dim = policy.config.action_dim

    policy.policy.eval()

    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    if policy_config["action_head"] == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif policy_config["action_head"] == 'transformer_diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    elif policy_config["action_head"] == 'droid_diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']

    query_frequency = policy.config.chunk_size / 2 # specify the exact executed action steps, must be smaller than chunk size
    if temporal_agg:
        query_frequency = 1
        num_queries = policy.config.chunk_size
    max_timesteps = int(200)  # may increase for real-world tasks


    ### evaluation loop
    if temporal_agg:
        all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, action_dim],dtype=torch.float32).cuda()
        # print(f'all_time_actions size: {all_time_actions.size()}')

    t = 0
    replay_traj = dict()
    image_list = []  # for visualization
    robot_state_list = []
    target_action_list = []
    success = False
    with torch.inference_mode():
        try:
            
            while t < cfg.num_steps_wait:
                # Do nothing for the first few timesteps to let objects stabilize
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue
            t = 0
            while t < max_timesteps:

                
                traj_rgb_np, robot_state = get_obs(obs=obs,
                                                   stats=stats)
                
                image_list.append(cv2.resize(traj_rgb_np[0], (256,256)))
                robot_state_list.append(robot_state)
                robot_state = torch.from_numpy(robot_state).float().cuda()

                if t % query_frequency == 0:
                    curr_image =  []
                    for img in traj_rgb_np:
                        curr_image.append(to_tensor(img).float().cuda())
                    curr_image = torch.stack(curr_image, dim=0)  # stack images along batch dimension
                    # curr_image = to_tensor(traj_rgb_np).float().cuda() #torch.from_numpy(traj_rgb_np / 255.0).float().cuda()
                    if rand_crop_resize:
                        # print('rand crop resize is used!')
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
                    # warm up
                    for _ in range(10):
                        batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                        policy.policy(**batch, eval=True)
                    # print('network warm up done')
                    time1 = time.time()

                ### query policy
                time3 = time.time()
                if policy_config['action_head'] == "act":
                    if t % query_frequency == 0:
                        batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                        all_actions = policy.policy(**batch, eval=True)

                    if temporal_agg:
                        # print(f"all_actions: {all_actions.size()}")
                        # print(f"all_time_actions: {all_time_actions.size()}")
                        # print(f"t: {t}, num_queries:{num_queries}")
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
                elif policy_config['action_head'] == "droid_diffusion":
                    if t % query_frequency == 0:
                        batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                        all_actions = policy.policy(**batch, eval=True)
                            
                    if temporal_agg:
                        # print(f"all_actions: {all_actions.size()}")
                        # print(f"all_time_actions: {all_time_actions.size()}")
                        # print(f"t: {t}, num_queries:{num_queries}")
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

                # print(f"raw action size: {raw_action.size()}")
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = raw_action
                action = post_process(raw_action)
                # print(f"after post_process action size: {action.shape}")
                
                
                action = convert_actions(action)
                # action = normalize_gripper_action(action=action, binarize=True)
                # action = invert_gripper_action(action)
                # print(f"after normalization and inversion: {action}")
                time5 = time.time()
                
                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                target_action_list.append(action)
                pil_img = Image.fromarray(obs['agentview_image'][::-1, ::-1])  # Convert to PIL image for saving
                pil_img.save(os.path.join(cfg.local_log_dir, f"step.png"))
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


def run_task(cfg: GenerateConfig, task_suite, task_id, policy, policy_config, log_file, total_episodes=0, total_successes=0):
    # Get task
    task = task_suite.get_task(task_id)

    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, 
                                           cfg.model_family, 
                                           resolution=cfg.env_img_res,
                                           change_spawn=cfg.change_spawn,
                                           train_spawn_distribution=cfg.spawn_train_distribution)

    # get the episode already recorded in the environment
    rollout_dir = f"./rollouts/{cfg.task_suite_name}/change_spawn_{cfg.change_spawn}_train_{cfg.spawn_train_distribution}/run_{cfg.run_number}"
    episode_full_list = glob.glob(os.path.join(rollout_dir, "*.npy"))
    len_episode_full_list = len(episode_full_list)
    episode_number = []
    for episode in episode_full_list:
        episode_number.append(int(episode.split("episode=")[-1].split("--")[0]))
    episode_number.sort()
    
    
    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        if total_episodes < len_episode_full_list:
            log_message(f"Skipping episode {total_episodes} as it already exists in {rollout_dir}", log_file)
            total_episodes += 1
            continue
        
        log_message(f"\nTask: {task_description}", log_file)
        # if episode_idx < len_episode_full_list:
        #     # If the episode already exists, skip it
        #     log_message(f"Skipping episode {episode_idx} as it already exists in {rollout_dir}", log_file)
        #     continue
        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode     
        if cfg.change_spawn:
            log_message("Setting initial state with changed spawn region...", log_file)
            initial_state, all_initial_state = None, None

        success, replay_traj = run_episode(
            cfg,
            env,
            task_description,
            policy,
            policy_config,
            224,
            initial_state,
            log_file,
        )
        
        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_traj, 
            total_episodes, 
            success=success, 
            task_description=task_description, 
            log_file=log_file,
            dataset_name=cfg.task_suite_name,
            run=cfg.run_number, 
            change_spawn=cfg.change_spawn,
            train_spawn_distribution=cfg.spawn_train_distribution
        )
        
        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)


    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes

@draccus.wrap()
def run_libero_eval(cfg: GenerateConfig):
    
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    action_head = 'droid_diffusion' # specify the action head type
    policy_config = {
        "model_path": f"{cfg.model_path}", # mainly includes the lora weights
        "model_base": f"{cfg.model_base}", # used for lora merge weights
        "enable_lora": True,
        "conv_mode": "pythia",
        "action_head": action_head,
    }
    
    # Set random seed
    set_seed_everywhere(seed=0)
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    
    # make policy
    policy = llava_pythia_act_policy(policy_config)

    
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    print(f"Evaluating {num_tasks} tasks in the {cfg.task_suite_name} task suite.")
    
    
    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg=cfg,
            task_suite=task_suite,
            task_id=task_id,
            policy=policy,
            policy_config=policy_config,
            log_file=log_file,
            total_episodes=total_episodes,
            total_successes=total_successes,
        )
    
    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    
    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == '__main__':
    run_libero_eval()
    
