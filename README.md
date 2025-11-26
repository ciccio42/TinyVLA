<h1 align="center">
TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models
for Robotic Manipulation</h1>

# Note
This is a fork from the original TinyVLA projecr

# Install

1. Clone this repository and navigate to diffusion-vla folder

2. Install Package
```Shell
conda create -n tinyvla python=3.10 -y
conda activate tinyvla
pip install --upgrade pip  # 
pip install -r requirements.txt
cd policy_heads
pip install -e . 
# install llava-pythia
cd ../llava-pythia
pip install -e . 
```

# VLA-Benchmark
**NOTE**
Update to 
torch-2.7.0+cu128, deepspeed-0.18.1, bitsandbytes-0.48.0
```
# in /.conda/envs/tinyvla/lib/python3.10/site-packages/transformers/trainer.py
import numpy
# Allowlist the necessary NumPy classes/functions
torch.serialization.add_safe_globals([
    numpy.core.multiarray._reconstruct,
    numpy.dtype
])
2239:checkpoint_rng_state = torch.load(rng_file, weights_only=False)

# in /.conda/envs/tinyvla/lib/python3.10/site-packages/transformers/integrations/deepspeed.py
402:load_path, _ = deepspeed_engine.load_checkpoint(
            checkpoint_path, 
            load_optimizer_states=True, 
            load_lr_scheduler_states=True,
            load_module_strict=False
        )
```

## Train

### 1. Download dataset and process datasets
Links to datasets will be available after paper acceptance
**NOTE**
In config file change the path to the dataset *dataset_dir*
```bash
cd data_utils
python ur5e_pick_place_rlds_to_h5py --cfg_path config/[DATASET-NAME].json --save_dir dataset
```

### 2. Run train
**NOTE**
In run_train.py set:
* **TASK_NAME** to desired task
* **CHECKPOINT_PATH** to checkpoint save folder
```bash
cd scripts
nohup python run_train.py
```

## Eval

### 1. Prepare checkpoint
```bash
cd scripts
# Inside script DATASET_NAME and min_step
sbatch process_ckpt.sh
# Inside script set DATASET_NAME
source copy_preprocessor.sh
```

### 2. Use VLA-Benchmark
See instruction [here]()



## Original Tiny-VLA Citation
This is a fork from the original repository of the following work.
```bibtex
@misc{
    @inproceedings{wen2024tinyvla,
    title={Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation},
    author={Wen, Junjie and Zhu, Yichen and Li, Jinming and Zhu, Minjie and Wu, Kun and Xu, Zhiyuan and Liu, Ning and Cheng, Ran and Shen, Chaomin and Peng, Yaxin and others},
    booktitle={IEEE Robotics and Automation Letters (RA-L)},
    year={2025}
}
```