DATASET_NAME=ur5e_pick_place_rm_one_spawn
LORA_R=128
OUTPUT=/home/rsofnc000/checkpoint_save_folder/tiny_vla/post_processed_tiny_vla_llava_pythia_lora_${DATASET_NAME}_lora_r_${LORA_R}_processed

for dir in "$OUTPUT"/*/ ; do
    # 检查文件夹名称是否包含'checkpoint'
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        echo "Coping preprocessor_config.json to $dir"
        cp preprocessor_config.json $dir
    fi
done
