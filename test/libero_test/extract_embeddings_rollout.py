"""
extract_embeddings_rollout.py

Extract pre-action-head embeddings from TinyVLA model during REAL inference rollouts.
Uses the model's forward pass to execute real actions and captures the hidden_states
(output of GPT-NeoX / Pythia backbone) BEFORE they are passed to the diffusion action head.
"""

import os
import sys
import torch
import pickle
import numpy as np
import cv2
import argparse
import gc
import logging
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(output_dir: str, task_suite: str):
    """Setup logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"extraction_log_{task_suite}_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger('TinyVLA_Extraction')
    logger.setLevel(logging.DEBUG)
    
    # File handler (detailed)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    
    # Console handler (less verbose)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(ch_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file

def log_gpu_memory(logger, stage: str):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        logger.debug(f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")
        return allocated, reserved
    return 0, 0

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"

from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
from llava_pythia.conversation import conv_templates, SeparatorStyle
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava_pythia.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
)
from llava_pythia.model import *

from torchvision import transforms
from einops import rearrange
import torch_utils as TorchUtils

from libero.libero import get_libero_path, benchmark
from libero.libero.envs import OffScreenRenderEnv

from libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    get_libero_env,
    quat2axisangle,
    extract_command_from_bddl,
)
from robot_utils import set_seed_everywhere

# Global logger (will be initialized in main)
logger = None

# ============================================================================
# Constants
# ============================================================================

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# ============================================================================
# Policy class
# ============================================================================

class llava_pythia_act_policy:
    """Policy class for Llava-Pythia action generation."""

    def __init__(self, policy_config, data_args=None):
        super(llava_pythia_act_policy).__init__()
        logger.info("=" * 60)
        logger.info("INITIALIZING POLICY")
        logger.info("=" * 60)
        self.load_policy(policy_config)
        self.data_args = data_args
        logger.info("Policy initialization complete")

    def load_policy(self, policy_config):
        global logger
        
        logger.info("Starting policy loading...")
        logger.info(f"Model path: {policy_config['model_path']}")
        logger.info(f"Model base: {policy_config.get('model_base', 'None')}")
        logger.info(f"LoRA enabled: {policy_config.get('enable_lora', False)}")
        
        # Clear any existing GPU memory
        logger.info("Clearing GPU cache before loading...")
        log_gpu_memory(logger, "Before cleanup")
        torch.cuda.empty_cache()
        gc.collect()
        log_gpu_memory(logger, "After cleanup")
        
        self.policy_config = policy_config
        model_base = policy_config["model_base"] if policy_config['enable_lora'] else None
        model_name = get_model_name_from_path(policy_config['model_path'])
        model_path = policy_config["model_path"]
        
        logger.info(f"Loading model: {model_name}")
        logger.info("Step 1/3: Loading tokenizer...")
        
        try:
            self.tokenizer, self.policy, self.image_processor, self.context_len = load_pretrained_model(
                model_path, model_base, model_name, False, False
            )
            logger.info("Step 2/3: Base model loaded successfully")
            log_gpu_memory(logger, "After base model load")
            
            # ADD THIS LINE - Load the config
            self.config = LlavaPythiaConfig.from_pretrained(
                '/'.join(model_path.split('/')[:-1]), trust_remote_code=True
            )
            logger.info("Step 3/3: Config loaded successfully")
            
            logger.info(f"Context length: {self.context_len}")
            logger.info(f"Action dimension: {self.config.action_dim}")
            logger.info(f"Chunk size: {self.config.chunk_size}")
            logger.info("Policy loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load policy: {str(e)}")
            log_gpu_memory(logger, "Error state")
            raise

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
# Helpers
# ============================================================================

def convert_actions(pred_action):
    """Convert 10D action (xyz + rot6d + gripper) to 7D (xyz + euler + gripper)."""
    cur_xyz = pred_action[:3]
    cur_rot6d = pred_action[3:9]
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)
    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)
    cur_euler = TorchUtils.rot_6d_to_euler_angles(rot_6d=cur_rot6d, convention="XYZ").squeeze().numpy()
    return np.concatenate((cur_xyz, cur_euler, cur_gripper))

def get_obs(obs, stats):
    """Retrieve and normalize observations from the environment."""
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

def build_bddl_path(task, level: str) -> str:
    """Return the BDDL path for default or syn_lX variation."""
    if level == "default":
        return os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    base_name = task.bddl_file.replace(".bddl", "")
    bddl_filename = f"{base_name}_syn_{level}.bddl"
    return os.path.join(get_libero_path("bddl_files"), task.problem_folder, bddl_filename)

# ============================================================================
# Embedding Extraction via Forward Hook
# ============================================================================

class EmbeddingCapture:
    """Captures hidden_states from the GPT-NeoX backbone via a forward hook."""

    def __init__(self):
        self.hidden_states = None
        self._handle = None
        logger.debug("EmbeddingCapture initialized")

    def hook_fn(self, module, input, output):
        """Capture the backbone output (hidden_states)."""
        hs = output[0]  # (B, seq_len, hidden_dim)
        self.hidden_states = hs.mean(dim=1).detach().cpu().float().numpy()

    def register(self, model):
        """Register the hook on the GPT-NeoX backbone."""
        backbone = model.get_model()
        self._handle = backbone.register_forward_hook(self.hook_fn)
        logger.info("Forward hook registered on GPT-NeoX backbone")

    def remove(self):
        """Remove the hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
            logger.debug("Forward hook removed")

    def get_embedding(self):
        """Return the last captured embedding as (hidden_dim,) numpy array."""
        if self.hidden_states is None:
            return None
        return self.hidden_states.squeeze(0)

# ============================================================================
# Episode Runner with Embedding Extraction
# ============================================================================

def run_episode_with_embeddings(
    env,
    task_description,
    policy,
    policy_config,
    stats,
    emb_capture,
    initial_state=None,
    max_steps=300,
    num_steps_wait=10,
    first_step_only=False,
):
    """Run a single episode and extract pre-action-head embeddings at each step."""
    
    logger.debug(f"Starting episode with task: '{task_description}'")
    logger.debug(f"Max steps: {max_steps}, First step only: {first_step_only}")
    
    env.reset()
    to_tensor = transforms.ToTensor()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
        logger.debug("Set initial state from checkpoint")
    else:
        obs = env.reset()
        logger.debug("Reset environment to default state")

    # Configuration
    action_dim = policy.config.action_dim
    policy.policy.eval()

    # Post-processing
    if policy_config["action_head"] == 'droid_diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    elif policy_config["action_head"] == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    else:
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']

    # Temporal aggregation parameters
    temporal_agg = True
    query_frequency = 1
    num_queries = policy.config.chunk_size
    max_timesteps = int(max_steps)

    if temporal_agg:
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + num_queries, action_dim],
            dtype=torch.float32
        ).cuda()

    embeddings = []
    t = 0
    success = False

    with torch.inference_mode():
        try:
            # Wait for stabilization
            logger.debug(f"Waiting {num_steps_wait} steps for environment stabilization...")
            while t < num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action("tiny_vla"))
                t += 1

            t = 0
            logger.debug("Starting main rollout loop...")
            
            while t < max_timesteps:
                # Log progress every 50 steps
                if t % 50 == 0 and t > 0:
                    logger.debug(f"Episode progress: {t}/{max_timesteps} steps")
                
                traj_rgb_np, robot_state = get_obs(obs=obs, stats=stats)
                robot_state = torch.from_numpy(robot_state).float().cuda()

                if t % query_frequency == 0:
                    curr_image = []
                    for img in traj_rgb_np:
                        curr_image.append(to_tensor(img).float().cuda())
                    curr_image = torch.stack(curr_image, dim=0)

                    # Random crop resize (for droid_diffusion)
                    if policy_config["action_head"] != 'act':
                        original_size = curr_image.shape[-2:]
                        ratio = 0.95
                        curr_image = curr_image[:, :,
                                        int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                                        int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                        curr_image = curr_image.squeeze(0)
                        resize_transform = transforms.Resize(original_size, antialias=True)
                        curr_image = resize_transform(curr_image)
                        curr_image = curr_image.unsqueeze(0)

                # Warmup on first step
                if t == 0:
                    logger.debug("Running warmup inference (10 iterations)...")
                    for warmup_iter in range(10):
                        batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                        policy.policy(**batch, eval=True)
                    logger.debug("Warmup complete")

                # Query policy (the forward hook captures hidden_states automatically)
                if t % query_frequency == 0:
                    batch = policy.process_batch_to_llava(curr_image, robot_state, task_description)
                    all_actions = policy.policy(**batch, eval=True)

                # Capture embedding from the hook
                embedding = emb_capture.get_embedding()
                if embedding is not None:
                    embeddings.append(embedding)
                    if t == 0:
                        logger.debug(f"First embedding captured, shape: {embedding.shape}")
                else:
                    logger.warning(f"No embedding captured at step {t}")

                # Return early if only first step needed
                if first_step_only:
                    logger.debug("First step extraction complete, returning early")
                    return embeddings, False, 1

                # Temporal aggregation
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

                # Post-process and execute
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action = convert_actions(action)

                obs, reward, done, info = env.step(action.tolist())

                if done:
                    success = True
                    logger.debug(f"Episode completed successfully at step {t}")
                    break
                t += 1

        except Exception as e:
            logger.error(f"Episode error at step {t}: {str(e)}")
            logger.exception("Full traceback:")

    logger.debug(f"Episode finished: {len(embeddings)} embeddings extracted, success={success}")
    return embeddings, success, len(embeddings)

# ============================================================================
# Main Extraction
# ============================================================================

def extract_embeddings_rollout(
    model_path: str,
    model_base: str,
    task_suite_name: str = "libero_goal",
    command_levels=("default", "l1", "l2", "l3"),
    output_dir: str = "/mnt/beegfs/a.cardamone7/outputs/embeddings/tinyvla",
    resolution: int = 256,
    seed: int = 0,
    num_rollouts_per_task: int = 10,
    first_step_only: bool = False,
):
    """Extract mean pre-action-head embeddings during real inference rollouts."""
    
    global logger
    
    # Setup logging
    logger, log_file = setup_logging(output_dir, task_suite_name)
    
    logger.info("=" * 80)
    logger.info("TINYVLA EMBEDDING EXTRACTION STARTED")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model base: {model_base}")
    logger.info(f"Task suite: {task_suite_name}")
    logger.info(f"Command levels: {command_levels}")
    logger.info(f"Rollouts per task: {num_rollouts_per_task}")
    logger.info(f"First step only: {first_step_only}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Resolution: {resolution}")
    logger.info("=" * 80)

    # ---- Load policy ----
    policy_config = {
        "model_path": model_path,
        "model_base": model_base,
        "enable_lora": True,
        "conv_mode": "pythia",
        "action_head": "droid_diffusion",
    }
    
    policy = llava_pythia_act_policy(policy_config)
    policy.policy.eval()
    logger.info("Policy set to eval mode")

    # Load dataset stats for normalization
    stats_path = os.path.join("/".join(model_path.split('/')[:-1]), 'dataset_stats.pkl')
    logger.info(f"Loading dataset stats from: {stats_path}")
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    logger.info("Dataset stats loaded successfully")

    # ---- Register embedding hook ----
    emb_capture = EmbeddingCapture()
    emb_capture.register(policy.policy)

    # ---- Task suite ----
    logger.info(f"Setting seed: {seed}")
    set_seed_everywhere(seed)
    
    logger.info(f"Loading task suite: {task_suite_name}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = TASK_MAX_STEPS.get(task_suite_name, 300)
    
    logger.info(f"Task suite loaded: {num_tasks} tasks, max {max_steps} steps per episode")

    mode_str = "FIRST STEP ONLY" if first_step_only else "FULL ROLLOUT"
    total_episodes = num_tasks * num_rollouts_per_task * len(command_levels)
    logger.info(f"Mode: {mode_str}")
    logger.info(f"Total episodes to run: {total_episodes}")
    logger.info("=" * 80)

    all_embeddings = {}
    episode_counter = 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_name = getattr(task, 'name', str(task))

        # Get initial states for this task
        initial_states = task_suite.get_task_init_states(task_id)
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"TASK {task_id + 1}/{num_tasks}: {task_name}")
        logger.info(f"Initial states available: {len(initial_states)}")
        logger.info("=" * 80)

        for level in command_levels:
            bddl_file = build_bddl_path(task, level)

            if not os.path.exists(bddl_file):
                logger.warning(f"  {level.upper():8s}: BDDL file not found: {bddl_file}")
                continue

            command = extract_command_from_bddl(bddl_file)
            if command is None:
                logger.warning(f"  {level.upper():8s}: Could not extract command")
                continue

            logger.info(f"\n  COMMAND LEVEL: {level.upper()}")
            logger.info(f"  Command text: '{command}'")
            logger.info(f"  Running {num_rollouts_per_task} rollouts...")

            # Collect embeddings from all rollouts
            rollout_embeddings = []
            rollout_all_embeddings = []
            rollout_successes = []
            successes = 0
            total_steps = 0

            for rollout_idx in range(num_rollouts_per_task):
                episode_counter += 1
                logger.info(f"\n  Rollout {rollout_idx+1}/{num_rollouts_per_task} "
                           f"(Episode {episode_counter}/{total_episodes})")
                
                # Create environment
                try:
                    logger.debug(f"    Creating environment...")
                    env, task_description, _ = get_libero_env(
                        task, "tiny_vla",
                        change_command=(level != "default"),
                        command_level=level if level != "default" else None,
                        resolution=resolution,
                    )
                    env.seed(seed + rollout_idx)
                    logger.debug(f"    Environment created, seed={seed + rollout_idx}")
                except Exception as e:
                    logger.error(f"    Failed to create environment: {e}")
                    continue

                # Get initial state
                init_state = initial_states[rollout_idx % len(initial_states)]
                logger.debug(f"    Using initial state {rollout_idx % len(initial_states)}")

                try:
                    episode_embeddings, success, num_steps = run_episode_with_embeddings(
                        env=env,
                        task_description=command,
                        policy=policy,
                        policy_config=policy_config,
                        stats=stats,
                        emb_capture=emb_capture,
                        initial_state=init_state,
                        max_steps=max_steps,
                        num_steps_wait=10,
                        first_step_only=first_step_only,
                    )

                    if episode_embeddings:
                        if first_step_only:
                            rollout_embeddings.append(episode_embeddings[0])
                            logger.info(f"    ✓ Embedding extracted (1 step)")
                        else:
                            rollout_emb = np.stack(episode_embeddings, axis=0)
                            rollout_mean = np.mean(rollout_emb, axis=0)
                            rollout_embeddings.append(rollout_mean)
                            rollout_all_embeddings.append(rollout_emb)
                            rollout_successes.append(success)

                            successes += int(success)
                            total_steps += num_steps

                            status = "✓ SUCCESS" if success else "✗ FAILURE"
                            logger.info(f"    {status} - {num_steps} steps, {len(episode_embeddings)} embeddings")
                    else:
                        logger.warning(f"    No embeddings extracted")

                except Exception as e:
                    logger.error(f"    Error during rollout: {e}")
                    logger.exception("    Full traceback:")
                finally:
                    try:
                        env.close()
                        logger.debug("    Environment closed")
                    except:
                        pass

            # Compute statistics
            if rollout_embeddings:
                rollout_embeddings_arr = np.stack(rollout_embeddings, axis=0)
                mean_embedding = np.mean(rollout_embeddings_arr, axis=0)

                key = f"task_{task_id:02d}_{level}"
                all_embeddings[key] = {
                    "task_id": task_id,
                    "task_name": task_name,
                    "command_level": level,
                    "command_text": command,
                    "embedding": mean_embedding,
                    "embedding_per_rollout": rollout_embeddings_arr,
                    "num_rollouts": len(rollout_embeddings),
                    "first_step_only": first_step_only,
                    "model": "tinyvla",
                }

                if not first_step_only and rollout_all_embeddings:
                    all_embeddings[key]["embedding_all_steps"] = np.concatenate(rollout_all_embeddings, axis=0)
                    all_embeddings[key]["rollout_successes"] = rollout_successes
                    all_embeddings[key]["num_successes"] = successes
                    all_embeddings[key]["total_steps"] = total_steps
                    all_embeddings[key]["success_rate"] = successes / max(len(rollout_embeddings), 1)

                if first_step_only:
                    logger.info(f"  Summary: {len(rollout_embeddings)} embeddings, "
                               f"shape: {mean_embedding.shape}")
                else:
                    success_rate = successes / max(len(rollout_embeddings), 1)
                    logger.info(f"  Summary: {successes}/{num_rollouts_per_task} success "
                               f"({success_rate:.1%}), {total_steps} total steps, "
                               f"embedding shape: {mean_embedding.shape}")
            else:
                logger.warning(f"  No embeddings extracted for {level}")

    # ---- Cleanup hook ----
    emb_capture.remove()
    logger.info("\nEmbedding capture hook removed")

    # ---- Save results ----
    logger.info(f"\nSaving results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    mode_suffix = "first_step" if first_step_only else "full"
    output_file = os.path.join(
        output_dir,
        f"rollout_embeddings_tinyvla_{task_suite_name}_{'_'.join(command_levels)}_{mode_suffix}_r{num_rollouts_per_task}.pkl"
    )

    logger.info(f"Writing to file: {output_file}")
    with open(output_file, "wb") as f:
        pickle.dump(all_embeddings, f)
    logger.info("Results saved successfully")

    # ---- Print summary ----
    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 80)

    if all_embeddings:
        first_key = next(iter(all_embeddings.keys()))
        logger.info(f"Total entries: {len(all_embeddings)}")
        logger.info(f"Mean embedding shape: {all_embeddings[first_key]['embedding'].shape}")
        logger.info(f"Mode: {'First step only' if first_step_only else 'Full rollout'}")

        if not first_step_only:
            logger.info("\nSuccess rates by command level:")
            for level in command_levels:
                level_data = [v for k, v in all_embeddings.items() if v['command_level'] == level]
                if level_data:
                    avg_sr = np.mean([d.get('success_rate', 0) for d in level_data])
                    avg_steps = np.mean([d.get('total_steps', 0) / d.get('num_rollouts', 1) for d in level_data])
                    logger.info(f"  {level.upper():8s}: {avg_sr:6.1%} success rate, {avg_steps:5.1f} avg steps")

    logger.info(f"\nOutput file: {output_file}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    return all_embeddings, output_file

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract pre-action-head embeddings during real inference rollouts from TinyVLA on LIBERO"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to TinyVLA checkpoint (e.g., .../checkpoint-54000)",
    )
    parser.add_argument(
        "--model_base",
        type=str,
        required=True,
        help="Path to base model (e.g., .../1.3B)",
    )
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_goal",
    )
    parser.add_argument(
        "--command_levels",
        type=str,
        nargs="+",
        default=["default", "l1", "l2", "l3"],
        help="Command levels to extract (e.g., --command_levels default l1 l2 l3)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/beegfs/a.cardamone7/outputs/embeddings/tinyvla",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=10,
        help="Number of rollout episodes per task (default: 10)",
    )
    parser.add_argument(
        "--first_step_only",
        action="store_true",
        help="Extract embedding only from first observation (no full rollout)",
    )

    args = parser.parse_args()

    extract_embeddings_rollout(
        model_path=args.model_path,
        model_base=args.model_base,
        task_suite_name=args.task_suite,
        command_levels=tuple(args.command_levels),
        output_dir=args.output_dir,
        resolution=args.resolution,
        seed=args.seed,
        num_rollouts_per_task=args.num_rollouts,
        first_step_only=args.first_step_only,
    )