import os
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import debugpy
import matplotlib.patches as patches
import glob
import pickle as pkl

TABLE_SIZE = (1.0, 1.0)  # height (y), width (x)
# Enable remote debugging
# debugpy.listen(('0.0.0.0', 5678)) 
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

def heat_map(task_distribution, task_path, task_name):
    # Each px represents 0.5x0.5 cm (0.005m x 0.005m)
    px_resolution = 0.5  # in cm

    # Convert TABLE_SIZE to cm and then to pixels
    table_size_cm = np.array(TABLE_SIZE) * 100  # meters to cm
    table_size_px = (table_size_cm / px_resolution).astype(np.int32)

    # Initialize table heatmap
    table_map = np.zeros((table_size_px[0], table_size_px[1]))  # shape: (height, width)

    # Loop through each episode's trajectory
    for episode_idx, trajectory in enumerate(task_distribution):
        print(f"Processing episode: {episode_idx}")

        # Convert list of [x, y, z] to np.array and take only x, y
        trajectory = np.array(trajectory)[:, :2]  # shape: (T, 2)

        # Convert meters to cm and to pixels
        px_traj = (trajectory * 100 / px_resolution).astype(np.int32)

        # Translate coords: center of table is the middle of the image
        px_traj[:, 0] = table_map.shape[0] // 2 + px_traj[:, 0]  # x -> vertical axis (rows)
        px_traj[:, 1] = table_map.shape[1] // 2 + px_traj[:, 1]  # y -> horizontal axis (cols)

        # Clip to table bounds
        px_traj = px_traj[
            (px_traj[:, 0] >= 0) & (px_traj[:, 0] < table_map.shape[0]) &
            (px_traj[:, 1] >= 0) & (px_traj[:, 1] < table_map.shape[1])
        ]

        # Populate heatmap
        for x, y in px_traj:
            table_map[x, y] += 1

    # Set crop range in cm for visual focus (adjust as needed)
    y_min, y_max = -45, 20
    x_min, x_max = -35, 35
    task_title = task_name.replace("_", " ").title()

    # Convert to pixel bounds
    y_min_px = int((y_min + table_size_cm[0] / 2) / px_resolution)
    y_max_px = int((y_max + table_size_cm[0] / 2) / px_resolution)
    x_min_px = int((x_min + table_size_cm[1] / 2) / px_resolution)
    x_max_px = int((x_max + table_size_cm[1] / 2) / px_resolution)

    # Crop table
    cropped_map = table_map[y_min_px:y_max_px, x_min_px:x_max_px]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 14))
    plt.title(f"Command: '{task_title}'")
    plt.xlabel("Y Axis (cm)")
    plt.ylabel("X Axis (cm)")

    norm = mcolors.LogNorm(vmin=1, vmax=np.max(cropped_map) if np.max(cropped_map) > 0 else 1)
    im = ax.imshow(cropped_map, cmap='plasma', origin='upper', norm=norm)
    # ax.invert_yaxis()  # <-- This flips the y-axis so (0,0) is bottom-left
    ax.invert_xaxis()  # Invert x-axis to match the coordinate system
    
    # Axis ticks (every 10 cm)
    ticks_x = np.arange(0, cropped_map.shape[1], int(10 / px_resolution))
    ticks_y = np.arange(0, cropped_map.shape[0], int(10 / px_resolution))
    tick_labels_x = np.arange(x_min, x_max, 10)
    tick_labels_y = np.arange(y_min, y_max, 10)

    plt.xticks(ticks_x, tick_labels_x)
    plt.yticks(ticks_y, tick_labels_y)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Trajectory Density (log scale)")

    # Save
    os.makedirs(task_path, exist_ok=True)
    save_path = os.path.join(task_path, f"{task_name}_heatmap.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap to {save_path}")



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Create heatmaps for test trajectories")
    argparser.add_argument('--debug', action='store_true', help="Enable remote debugging")
    args = argparser.parse_args()
    
    if args.debug:
        # Enable remote debugging
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
    
    
    test_path = "/home/A.CARDAMONE7/outputs/rollouts/libero_goal/tinyvla/l1"
    dataset_config_file = "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/checkpoints_saving_folder/tinyvla/parte2_tiny_vla_llava_pythia_lora_libero_goal_no_noops_lora_r_64/dataset_stats.pkl"
    dataset_config = pkl.load(open(dataset_config_file, "rb"))
                              
    run_folders = glob.glob(os.path.join(test_path, "run_*"))

    tasks_trajectories = {}


    for run in run_folders:
        trajectories_npy = glob.glob(os.path.join(run, "*.npy"))

        trajectories_npy.sort(key=lambda x: int(os.path.basename(x).split("episode=")[-1].split("--")[0]))

        for trajectory_npy in trajectories_npy:
            data = np.load(trajectory_npy, allow_pickle=True).item()
            print(f"Analyzing {trajectory_npy.split('/')[-1]}...")
    
            task_name = data['task_command']
            print(f"Task command: {task_name}")
            if task_name not in tasks_trajectories:
                tasks_trajectories[task_name] = []
                
            states = np.array(data['states'])[:, :3]
            
            # denormalize positions
            qpos_mean = dataset_config['qpos_mean'][:3]
            gpos_std = dataset_config['qpos_std'][:3]
            states = (states * gpos_std) + qpos_mean
            
            tasks_trajectories[task_name].append(states)  # list of [x, y, z]
            
    # Create heatmaps for each task
    for task_name, episodes in tasks_trajectories.items():
        print(f"Creating heatmap for task: {task_name} with {len(episodes)} episodes")
        heat_map(episodes, test_path, task_name)
        