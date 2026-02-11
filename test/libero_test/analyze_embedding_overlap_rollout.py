"""
analyze_embedding_overlap_rollout.py

Analyze command variations using TinyVLA rollout-based pre-action-head embeddings:
- Cosine Similarity (semantic - mean embeddings from rollouts)
- Levenshtein Distance (lexical)

Embeddings are hidden_states from GPT-NeoX (Pythia) backbone, mean-pooled over the
sequence dimension, captured BEFORE the diffusion action head.

Supports:
- first_step_only mode: one embedding per rollout from first observation
- full rollout mode: mean embedding per rollout from all steps
"""

import os
import glob
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import pandas as pd
from Levenshtein import distance as levenshtein_distance


def compute_levenshtein_normalized(text1, text2):
    """
    Compute normalized Levenshtein distance between two texts.
    
    Args:
        text1: First sentence (string)
        text2: Second sentence (string)
    
    Returns:
        Normalized Levenshtein distance (0-1, where 0=identical)
    """
    lev_dist = levenshtein_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 0.0
    return lev_dist / max_len


def load_embeddings(embedding_files=None, embedding_dir=None):
    """
    Load embeddings from multiple files or a directory.
    
    Args:
        embedding_files: List of pickle file paths
        embedding_dir: Directory containing pickle files
        
    Returns:
        Combined dictionary of all embeddings
    """
    all_embeddings = {}
    
    # Determine files to load
    if embedding_dir:
        files = sorted(glob.glob(os.path.join(embedding_dir, "*.pkl")))
        print(f"Found {len(files)} pickle files in {embedding_dir}")
    elif embedding_files:
        files = embedding_files
    else:
        raise ValueError("Must provide embedding_files or embedding_dir")
    
    # Load and merge all files
    for filepath in files:
        print(f"  Loading: {os.path.basename(filepath)}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            all_embeddings.update(data)
    
    print(f"\nTotal embeddings loaded: {len(all_embeddings)}")
    return all_embeddings


def analyze_embeddings(embeddings, output_csv="analysis_results.csv"):
    """
    Analyze rollout-based mean embeddings with Cosine similarity and Levenshtein distance.
    
    Args:
        embeddings: Dictionary of embeddings (already loaded)
        output_csv: Path for output CSV file
    
    Uses mean embeddings computed over rollout steps for more robust semantic similarity.
    """
    
    print(f"\nAnalyzing {len(embeddings)} rollout-based mean embeddings\n")
    
    # Print info about first embedding
    if embeddings:
        first_key = next(iter(embeddings.keys()))
        first_data = embeddings[first_key]
        print(f"  Model: {first_data.get('model', 'unknown')}")
        print(f"  Mean embedding shape: {first_data['embedding'].shape}")
        print(f"  Embeddings per rollout shape: {first_data['embedding_per_rollout'].shape}")
        print(f"  Number of rollouts: {first_data.get('num_rollouts', 'N/A')}")
        mode = "First step only" if first_data.get('first_step_only', False) else "Full rollout"
        print(f"  Mode: {mode}")
        if 'total_steps' in first_data:
            print(f"  Total steps: {first_data['total_steps']}")
        if 'success_rate' in first_data:
            print(f"  Success rate: {first_data['success_rate']:.2%}")
        print()
    
    # Organize by task
    tasks = {}
    for key, data in embeddings.items():
        task_id = data['task_id']
        if task_id not in tasks:
            tasks[task_id] = {}
        tasks[task_id][data['command_level']] = data
    
    results = []
    
    print("="*80)
    print("DISTANCE ANALYSIS: Semantic (Cosine) + Lexical (Levenshtein)")
    print("Using ROLLOUT MEAN EMBEDDINGS (pre-action-head hidden_states)")
    print("="*80)
    
    for task_id in sorted(tasks.keys()):
        task_data = tasks[task_id]
        
        if 'default' not in task_data:
            continue
        
        # Handle both 1D and 2D embeddings
        default_emb = task_data['default']['embedding']
        if default_emb.ndim > 1:
            default_emb = default_emb.flatten()
        default_cmd = task_data['default']['command_text']
        default_rollouts = task_data['default'].get('num_rollouts', 'N/A')
        default_sr = task_data['default'].get('success_rate', None)
        
        print(f"\nTask {task_id}")
        sr_str = f", SR={default_sr:.0%}" if default_sr is not None else ""
        print(f"  Default: {default_cmd} [{default_rollouts} rollouts{sr_str}]")
        
        for level in ['l1', 'l2', 'l3']:
            if level not in task_data:
                print(f"  {level.upper():3s}:     [NOT FOUND]")
                continue
            
            var_emb = task_data[level]['embedding']
            if var_emb.ndim > 1:
                var_emb = var_emb.flatten()
            var_cmd = task_data[level]['command_text']
            var_rollouts = task_data[level].get('num_rollouts', 'N/A')
            var_sr = task_data[level].get('success_rate', None)
            
            # ===== METRICS =====
            
            # 1. SEMANTIC: Cosine Similarity on mean rollout embeddings
            cos_sim = cosine_similarity([default_emb], [var_emb])[0, 0]
            
            # 2. SEMANTIC: Euclidean Distance (L2) on mean rollout embeddings
            euc_dist = euclidean_distances([default_emb], [var_emb])[0, 0]
            
            # 3. SEMANTIC: Manhattan Distance (L1) on mean rollout embeddings
            man_dist = manhattan_distances([default_emb], [var_emb])[0, 0]
            
            # 4. LEXICAL: Normalized Levenshtein Distance
            lev_dist = compute_levenshtein_normalized(default_cmd, var_cmd)
            
            # Store results
            results.append({
                'task_id': task_id,
                'level': level,
                'default_command': default_cmd,
                'variation_command': var_cmd,
                'default_rollouts': default_rollouts,
                'variation_rollouts': var_rollouts,
                'default_success_rate': default_sr,
                'variation_success_rate': var_sr,
                'cosine_similarity': cos_sim,
                'euclidean_distance': euc_dist,
                'manhattan_distance': man_dist,
                'levenshtein_distance': lev_dist,
            })
            
            sr_str = f", SR={var_sr:.0%}" if var_sr is not None else ""
            print(f"  {level.upper():3s}:     {var_cmd} [{var_rollouts} rollouts{sr_str}]")
            print(f"           Semantic:   Cosine_sim={cos_sim:.4f}")
            print(f"           Semantic:   Euclidean_dist(L2)={euc_dist:.4f}")
            print(f"           Semantic:   Manhattan_dist(L1)={man_dist:.4f}")
            print(f"           Lexical:    Lev_dist={lev_dist:.4f}")
    
    df = pd.DataFrame(results)
    
    # ===== STATISTICS =====
    print("\n" + "="*80)
    print("AVERAGE STATISTICS BY LEVEL")
    print("="*80)
    
    for level in ['l1', 'l2', 'l3']:
        level_data = df[df['level'] == level]
        if len(level_data) == 0:
            continue
        
        print(f"\n{level.upper()}:")
        print(f"  Cosine Similarity (semantic):     {level_data['cosine_similarity'].mean():.4f} ± {level_data['cosine_similarity'].std():.4f}")
        print(f"  Euclidean Distance L2 (semantic): {level_data['euclidean_distance'].mean():.4f} ± {level_data['euclidean_distance'].std():.4f}")
        print(f"  Manhattan Distance L1 (semantic): {level_data['manhattan_distance'].mean():.4f} ± {level_data['manhattan_distance'].std():.4f}")
        print(f"  Levenshtein Distance (lexical):   {level_data['levenshtein_distance'].mean():.4f} ± {level_data['levenshtein_distance'].std():.4f}")
    
    # ===== OVERALL STATISTICS =====
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"  Overall Cosine Similarity:      {df['cosine_similarity'].mean():.4f} ± {df['cosine_similarity'].std():.4f}")
    print(f"  Overall Euclidean Distance L2:  {df['euclidean_distance'].mean():.4f} ± {df['euclidean_distance'].std():.4f}")
    print(f"  Overall Manhattan Distance L1:  {df['manhattan_distance'].mean():.4f} ± {df['manhattan_distance'].std():.4f}")
    print(f"  Overall Levenshtein Distance:   {df['levenshtein_distance'].mean():.4f} ± {df['levenshtein_distance'].std():.4f}")
    
    # ===== SAVE RESULTS =====
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved: {output_csv}")
    
    return df


def compare_per_rollout_similarity(embeddings, task_id=0, level='l1'):
    """
    Analyze cosine similarity across rollouts.
    
    Args:
        embeddings: Dictionary of embeddings (already loaded)
        task_id: Task ID to analyze
        level: Command level to compare against default
    
    Compares default vs variation embedding for each rollout.
    """
    default_key = f"task_{task_id:02d}_default"
    var_key = f"task_{task_id:02d}_{level}"
    
    if default_key not in embeddings or var_key not in embeddings:
        print(f"Keys not found: {default_key} or {var_key}")
        return None
    
    default_data = embeddings[default_key]
    var_data = embeddings[var_key]
    
    if 'embedding_per_rollout' not in default_data or 'embedding_per_rollout' not in var_data:
        print("Per-rollout embeddings not available")
        return None
    
    default_rollouts = default_data['embedding_per_rollout']
    var_rollouts = var_data['embedding_per_rollout']
    
    # Compute similarity for each pair of rollouts
    min_rollouts = min(len(default_rollouts), len(var_rollouts))
    similarities = []
    
    for i in range(min_rollouts):
        sim = cosine_similarity([default_rollouts[i]], [var_rollouts[i]])[0, 0]
        similarities.append(sim)
    
    similarities = np.array(similarities)
    
    # Compute per-rollout Euclidean distances (L2)
    euc_dists = []
    for i in range(min_rollouts):
        dist = euclidean_distances([default_rollouts[i]], [var_rollouts[i]])[0, 0]
        euc_dists.append(dist)
    
    euc_dists = np.array(euc_dists)
    
    # Compute per-rollout Manhattan distances (L1)
    man_dists = []
    for i in range(min_rollouts):
        dist = manhattan_distances([default_rollouts[i]], [var_rollouts[i]])[0, 0]
        man_dists.append(dist)
    
    man_dists = np.array(man_dists)
    
    print(f"\nTask {task_id} - Default vs {level.upper()}")
    print(f"  Default command: {default_data['command_text']}")
    print(f"  Variation command: {var_data['command_text']}")
    print(f"\n  Per-rollout cosine similarity ({min_rollouts} rollouts):")
    print(f"    Mean:   {similarities.mean():.4f}")
    print(f"    Std:    {similarities.std():.4f}")
    print(f"    Min:    {similarities.min():.4f}")
    print(f"    Max:    {similarities.max():.4f}")
    print(f"\n  Per-rollout euclidean distance L2 ({min_rollouts} rollouts):")
    print(f"    Mean:   {euc_dists.mean():.4f}")
    print(f"    Std:    {euc_dists.std():.4f}")
    print(f"    Min:    {euc_dists.min():.4f}")
    print(f"    Max:    {euc_dists.max():.4f}")
    print(f"\n  Per-rollout manhattan distance L1 ({min_rollouts} rollouts):")
    print(f"    Mean:   {man_dists.mean():.4f}")
    print(f"    Std:    {man_dists.std():.4f}")
    print(f"    Min:    {man_dists.min():.4f}")
    print(f"    Max:    {man_dists.max():.4f}")
    
    return similarities, euc_dists, man_dists


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze TinyVLA command variations with rollout-based pre-action-head embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python analyze_embedding_overlap_rollout.py --embedding_file embeddings.pkl

  # Multiple files (one per command level)
  python analyze_embedding_overlap_rollout.py --embedding_files \\
      /mnt/beegfs/a.cardamone7/outputs/embeddings/tinyvla/rollout_embeddings_tinyvla_libero_goal_default_first_step_r10.pkl \\
      /mnt/beegfs/a.cardamone7/outputs/embeddings/tinyvla/rollout_embeddings_tinyvla_libero_goal_l1_first_step_r10.pkl \\
      /mnt/beegfs/a.cardamone7/outputs/embeddings/tinyvla/rollout_embeddings_tinyvla_libero_goal_l2_first_step_r10.pkl \\
      /mnt/beegfs/a.cardamone7/outputs/embeddings/tinyvla/rollout_embeddings_tinyvla_libero_goal_l3_first_step_r10.pkl

  # From directory
  python analyze_embedding_overlap_rollout.py --embedding_dir /mnt/beegfs/a.cardamone7/outputs/embeddings/tinyvla
"""
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        default=None,
        help="Path to a single embeddings pickle file"
    )
    parser.add_argument(
        "--embedding_files",
        type=str,
        nargs="+",
        default=None,
        help="Paths to multiple embeddings pickle files (one per command level)"
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default=None,
        help="Directory containing embeddings pickle files"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path (default: auto-generated)"
    )
    parser.add_argument(
        "--per_rollout_analysis",
        action="store_true",
        help="Run per-rollout similarity analysis for a specific task"
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="Task ID for per-rollout analysis"
    )
    parser.add_argument(
        "--level",
        type=str,
        default="l1",
        help="Command level for per-rollout analysis"
    )
    
    args = parser.parse_args()
    
    # Determine how to load embeddings
    if args.embedding_file:
        # Single file mode
        with open(args.embedding_file, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded {len(embeddings)} embeddings from {args.embedding_file}")
        output_base = args.embedding_file
    elif args.embedding_files or args.embedding_dir:
        # Multiple files or directory mode
        embeddings = load_embeddings(
            embedding_files=args.embedding_files,
            embedding_dir=args.embedding_dir
        )
        output_base = args.embedding_dir or os.path.dirname(args.embedding_files[0])
        output_base = os.path.join(output_base, "combined_analysis")
    else:
        # Default: TinyVLA embeddings directory
        default_dir = "/mnt/beegfs/a.cardamone7/outputs/embeddings/tinyvla"
        embeddings = load_embeddings(embedding_dir=default_dir)
        output_base = os.path.join(default_dir, "combined_analysis")
    
    # Run analysis
    df = analyze_embeddings(embeddings, output_csv=args.output_csv or f"{output_base}.csv")
    
    if args.per_rollout_analysis:
        similarities = compare_per_rollout_similarity(
            embeddings, 
            task_id=args.task_id, 
            level=args.level
        )
