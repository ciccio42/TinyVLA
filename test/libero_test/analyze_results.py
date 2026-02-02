import os
import json
import argparse
import glob
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze results from robosuite experiments.")
    parser.add_argument("--path", type=str, required=True, help="Path to the results directory containing run folders.")
    args = parser.parse_args()

    result_folders = glob.glob(os.path.join(args.path, f"run_*"))
    
    result = dict()
    for result_folder in result_folders:
        run_number = result_folder.split('_')[-1]
        print(f"Processing folder: {result_folder} (Run {run_number})")
        result[run_number] = []
        
        npy_files = glob.glob(os.path.join(result_folder, "*.npy"))
        # print(npy_files)
        for file in npy_files:
            if "success=True" in file:
                result[run_number].append(1)
            elif "success=False" in file:
                result[run_number].append(0)
        
        print(f"Run {run_number} - Success Count: {sum(result[run_number])}, Total Trials: {len(result[run_number])}")    
        result = {k: np.mean(v) for k, v in result.items()}
        
    print("Results:")
    for run, success_rate in result.items():
        print(f"Run {run}: Success Rate = {round(success_rate, 3)}")
        
    # average success rate across all runs
    overall_success_rate = round(np.mean(list(result.values())), 3)
    overall_std_dev = round(np.std(list(result.values())), 3)
    print(f"Overall Success Rate: {overall_success_rate:.3f} ± {overall_std_dev:.3f}")
    result["overall"] = overall_success_rate
    result["std_dev"] = overall_std_dev
    # Save results to a JSON file
    output_file = os.path.join(args.path, "success_rates.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)