"""
verify_visual_embedding_tinyvla.py

Verifica se l'embedding visivo (CLIP-ViT-L/14-336 → mm_projector)
è simile tra i 10 task di LIBERO Goal per il modello TinyVLA (checkpoint 20000).

Identico come obiettivo a verify_visual_embedding.py di OpenVLA-OFT:
se i projected_patches sono quasi identici tra task diversi, il modello
deve affidarsi quasi esclusivamente al testo per distinguere le azioni.

Output:
  - Matrice 10x10 di cosine similarity tra visual embedding dei task (token_cat)
  - Matrice 10x10 di euclidean distance
  - Sanity check: stessa immagine + prompt diversi → vision features identiche?
"""

import os
import sys
import pickle
import numpy as np
import cv2
import torch
from pathlib import Path
from torchvision import transforms
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ─────────────────── path setup ───────────────────
# This script lives in TinyVLA/test/libero_test
# Add TinyVLA root (for llava_pythia, policy_heads, torch_utils, etc.)
# and LIBERO root (for libero.libero.benchmark)
SCRIPT_DIR  = Path(__file__).resolve().parent                       # test/libero_test
TINYVLA_ROOT = SCRIPT_DIR.parent.parent                              # TinyVLA/
LIBERO_ROOT  = TINYVLA_ROOT.parent.parent / "LIBERO"                # robosuite_test/LIBERO

for p in [str(TINYVLA_ROOT), str(SCRIPT_DIR), str(LIBERO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import get_model_name_from_path
from llava_pythia.model import *                                      # noqa: F401,F403 (registers model)
from libero.libero import benchmark
from libero_utils import (
    get_libero_env,
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
)

# ─────────────────── constants ───────────────────
CHECKPOINT_PATH = (
    "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/"
    "checkpoints_saving_folder/tinyvla/"
    "post_processed_tiny_vla_llava_pythia_lora_libero_goal_no_noops_lora_r_64_processed/"
    "checkpoint-20000"
)
MODEL_BASE = (
    "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/"
    "checkpoints_saving_folder/tinyvla/"
    "parte2_llava_pythia_libero_goal_no_noops_64/1.3B"
)

# LIBERO Goal task ordering (from tasks_info.txt, 0-indexed)
TASK_NAMES = [
    "put_the_wine_bottle_on_top_of_the_cabinet",     # task 0
    "open_the_top_drawer_and_put_the_bowl_inside",   # task 1
    "turn_on_the_stove",                             # task 2
    "put_the_bowl_on_top_of_the_cabinet",            # task 3
    "put_the_bowl_on_the_plate",                     # task 4
    "put_the_wine_bottle_on_the_rack",               # task 5
    "put_the_cream_cheese_in_the_bowl",              # task 6
    "open_the_middle_drawer_of_the_cabinet",         # task 7
    "push_the_plate_to_the_front_of_the_stove",      # task 8
    "put_the_bowl_on_the_stove",                     # task 9
]

# Same-image different-prompt strings for sanity check
SANITY_PROMPTS = [
    "put the wine bottle on the top of the drawer",
    "open the middle layer of the drawer",
    "put the bowl on the stove",
    "place the wine bottle on the top of the drawer",
]

NUM_STEPS_WAIT = 10
ENV_IMG_RES    = 256
MODEL_FAMILY   = "tiny_vla"


# ─────────────────── model loading ───────────────────
def load_model():
    model_name = get_model_name_from_path(CHECKPOINT_PATH)
    print(f"Loading TinyVLA model: {model_name}")
    print(f"  model_path: {CHECKPOINT_PATH}")
    print(f"  model_base: {MODEL_BASE}")

    tokenizer, policy, image_processor, context_len = load_pretrained_model(
        CHECKPOINT_PATH, MODEL_BASE, model_name, False, False
    )
    policy.eval()
    print(f"✓ Model loaded  (hidden_size={policy.config.hidden_size})")
    print(f"  visual_concat = {policy.config.concat}")
    print(f"  action_head   = {policy.config.action_head_type}")
    return policy, image_processor


# ─────────────────── image preprocessing ───────────────────
# Replicates the exact pipeline used in eval_libero.py for droid_diffusion
# (rand_crop_resize = True)

_TO_TENSOR = transforms.ToTensor()
_RAND_CROP_RATIO = 0.95


def preprocess_image_pair(
    img_np: np.ndarray,
    wrist_np: np.ndarray,
    image_processor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replicates the preprocessing in eval_libero.process_batch_to_llava:
      1. Resize numpy (uint8) → (320, 180)
      2. to_tensor → [0,1] float
      3. rand_crop_resize at 95%
      4. expand2square (pad to square with CLIP mean background)
      5. image_processor.preprocess(do_normalize=True, do_rescale=False)

    Returns:
        image_tensor   : (1, 3, 336, 336) on cuda
        image_tensor_r : (1, 3, 336, 336) on cuda
    """
    device = next(iter(image_processor.image_mean.__class__.__mro__),  # duck-type
                  None)

    def _to_tensor_and_crop(np_img: np.ndarray) -> torch.Tensor:
        """uint8 (H,W,3) → float tensor (3,H,W) with rand_crop_resize."""
        resized = cv2.resize(np_img, (320, 180))
        t = _TO_TENSOR(resized).float()         # (3, 180, 320), [0,1]

        # rand_crop_resize — same as eval_libero.py
        orig_h, orig_w = t.shape[-2], t.shape[-1]  # 180, 320
        ratio = _RAND_CROP_RATIO
        t_crop = t[
            :,
            int(orig_h * (1 - ratio) / 2): int(orig_h * (1 + ratio) / 2),
            int(orig_w * (1 - ratio) / 2): int(orig_w * (1 + ratio) / 2),
        ]
        resize_tf = transforms.Resize((orig_h, orig_w), antialias=True)
        t_out = resize_tf(t_crop)               # (3, 180, 320)
        return t_out

    img_t    = _to_tensor_and_crop(img_np)    # (3, 180, 320)
    wrist_t  = _to_tensor_and_crop(wrist_np)  # (3, 180, 320)

    # Add batch dim → (1, 3, 180, 320) then expand2square → (1, 320, 320, 3)
    bg_color = tuple(float(x) for x in image_processor.image_mean)

    def _expand2square(t: torch.Tensor) -> torch.Tensor:
        """(3, H, W) → (1, max_dim, max_dim, 3) with background padding."""
        t = t.unsqueeze(0)                          # (1, 3, H, W)
        _, c, h, w = t.shape
        max_dim = max(h, w)
        expanded = np.full((1, max_dim, max_dim, c), bg_color, dtype=np.float32)
        if h == w:
            expanded = t.permute(0, 2, 3, 1).cpu().numpy()
        elif h > w:
            offset = (max_dim - w) // 2
            expanded[:, :h, offset:offset + w, :] = t.permute(0, 2, 3, 1).cpu().numpy()
        else:
            offset = (max_dim - h) // 2
            expanded[:, offset:offset + h, :w, :] = t.permute(0, 2, 3, 1).cpu().numpy()
        return torch.tensor(expanded, dtype=t.dtype)  # (1, max_dim, max_dim, 3)

    img_sq    = _expand2square(img_t)    # (1, 320, 320, 3)
    wrist_sq  = _expand2square(wrist_t)  # (1, 320, 320, 3)

    def _preprocess(sq: torch.Tensor) -> torch.Tensor:
        pv = image_processor.preprocess(
            sq,
            return_tensors="pt",
            do_normalize=True,
            do_rescale=False,
            do_center_crop=False,
        )["pixel_values"]                           # (1, 3, 336, 336)
        return pv.to("cuda", dtype=torch.float32)

    return _preprocess(img_sq), _preprocess(wrist_sq)


# ─────────────────── visual embedding extraction ───────────────────
def extract_visual_embedding(
    policy,
    image_processor,
    img_np: np.ndarray,
    wrist_np: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estrae l'embedding visivo (CLIP → projector → token_cat) per una coppia di immagini.

    Pipeline:
      primary_np   → preprocess → image_tensor   → vision_tower → mm_projector → projected      (1, 576, 2048)
      wrist_np     → preprocess → image_tensor_r → vision_tower → mm_projector → projected_r    (1, 576, 2048)
      combined = cat([projected, projected_r], dim=1)                                           (1, 1152, 2048)
      patches_flat = combined.squeeze(0)    → (1152, 2048)
      patches_mean = patches_flat.mean(0)  → (2048,)

    Returns:
        patches_flat : (n_patches_total, hidden_dim)  — raw combined patches
        patches_mean : (hidden_dim,)                  — mean-pool
    """
    with torch.no_grad():
        img_t, wrist_t = preprocess_image_pair(img_np, wrist_np, image_processor)

        vision_tower = policy.get_model().get_vision_tower()
        mm_projector  = policy.get_model().mm_projector

        # Primary image
        feats   = vision_tower(img_t)                # (1, n_patches, 1024)
        proj    = mm_projector(feats)                # (1, n_patches, 2048)

        # Wrist image
        feats_r = vision_tower(wrist_t)              # (1, n_patches, 1024)
        proj_r  = mm_projector(feats_r)             # (1, n_patches, 2048)

        # token_cat (as in get_image_fusion_embedding with visual_concat='token_cat')
        combined = torch.cat([proj, proj_r], dim=1)  # (1, 2*n_patches, 2048)

        patches_flat = combined.squeeze(0).detach().cpu().float().numpy()  # (2*n_patches, 2048)
        patches_mean = patches_flat.mean(axis=0)                           # (2048,)

    return patches_flat, patches_mean


# ─────────────────── first frame helper ───────────────────
def get_first_frame(
    task,
    task_id: int,
    task_suite,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inizializza l'ambiente LIBERO per il task dato, aspetta la stabilizzazione
    e restituisce il primo frame come numpy uint8 arrays (primary + wrist).
    """
    env, _, _ = get_libero_env(task, MODEL_FAMILY, change_command=False, resolution=ENV_IMG_RES)
    env.seed(0)

    try:
        initial_states = task_suite.get_task_init_states(task_id)
        env.reset()
        obs = env.set_init_state(initial_states[0])
    except Exception:
        env.reset()
        obs = env.get_observation()

    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action(MODEL_FAMILY))

    img_np    = get_libero_image(obs)       # (H, W, 3) uint8, flipped
    wrist_np  = get_libero_wrist_image(obs) # (H, W, 3) uint8, flipped
    env.close()

    return img_np, wrist_np


# ─────────────────── distance utilities ───────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ─────────────────── pretty print matrix ───────────────────
def short(name: str, n: int = 18) -> str:
    words = name.replace("_", " ").split()
    abbr = " ".join(w[:4] for w in words)
    return abbr[:n]


def print_matrix(matrix: np.ndarray, labels: List[str], title: str, fmt: str = "{:.4f}"):
    n = len(labels)
    col_w, lbl_w = 10, 22
    width = n * col_w + lbl_w + 4
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")
    header = f"{'':>{lbl_w}}"
    for i in range(n):
        header += f" {i:>{col_w-1}}"
    print(header)
    for i in range(n):
        row = f"  {i:2d} {short(labels[i]):<{lbl_w-4}}"
        for j in range(n):
            row += f" {fmt.format(matrix[i, j]):>{col_w-1}}"
        print(row)


# ─────────────────── sanity check ───────────────────
def sanity_check_text_independence(
    policy,
    image_processor,
    img_np: np.ndarray,
    wrist_np: np.ndarray,
):
    """
    Verifica che lo stesso frame con prompt diversi produca projected_patches identici.
    In TinyVLA, il vision_tower (CLIP) non opera su testo → risultato SEMPRE identico.
    """
    print("\n" + "=" * 80)
    print("SANITY CHECK: stessa immagine → visual embedding identici per ogni run?")
    print("(Il CLIP encoder non riceve mai testo: il risultato deve essere bit-for-bit uguale)")
    print("=" * 80)

    results = []
    for run_idx in range(len(SANITY_PROMPTS)):
        flat, _ = extract_visual_embedding(policy, image_processor, img_np, wrist_np)
        results.append(flat)
        print(f"  run {run_idx+1:2d}: patches shape = {flat.shape}")

    ref = results[0]
    all_identical = True
    for i, flat in enumerate(results[1:], 1):
        max_diff  = float(np.abs(ref - flat).max())
        identical = np.allclose(ref, flat, atol=1e-5)
        all_identical = all_identical and identical
        status = "✓ IDENTICI" if identical else "✗ DIVERSI"
        print(f"\n  Run 0 vs Run {i}: {status}  max|diff| = {max_diff:.2e}")

    if all_identical:
        print("\n  ✓ CONFERMATO: il blocco visivo è deterministico e testo-indipendente")
    else:
        print("\n  ✗ ATTENZIONE: risultati non identici tra run — verificare dropout/BN in eval mode")


# ─────────────────── main ───────────────────
def main():
    policy, image_processor = load_model()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict["libero_goal"]()
    n_tasks        = task_suite.n_tasks  # 10

    print(f"\n{'='*80}")
    print(f"VISUAL EMBEDDING ANALYSIS — TinyVLA checkpoint 20000 — LIBERO Goal")
    print(f"{'='*80}")
    print(f"visual_concat  = {policy.config.concat}")
    print(f"CLIP hidden    = 1024  →  projector output (hidden_size) = {policy.config.hidden_size}")
    print(f"Num tasks      = {n_tasks}")
    print(f"{'='*80}\n")

    # ── Estrai mean visual embedding per ogni task ──
    mean_embeddings: List[np.ndarray] = []
    flat_embeddings: List[np.ndarray] = []
    task_labels:     List[str] = []

    for task_id in range(n_tasks):
        task      = task_suite.get_task(task_id)
        task_key  = TASK_NAMES[task_id]
        task_labels.append(task_key)

        print(f"[{task_id:2d}] {task_key}")
        img_np, wrist_np = get_first_frame(task, task_id, task_suite)

        flat, mean = extract_visual_embedding(policy, image_processor, img_np, wrist_np)
        flat_embeddings.append(flat)
        mean_embeddings.append(mean)
        print(f"     combined patches shape: {flat.shape}  |  mean shape: {mean.shape}")

    # ── Matrici di similarità ──
    n = n_tasks
    cos_matrix = np.zeros((n, n))
    euc_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            cos_matrix[i, j] = cosine_sim(mean_embeddings[i], mean_embeddings[j])
            euc_matrix[i, j] = euclidean_dist(mean_embeddings[i], mean_embeddings[j])

    print_matrix(
        cos_matrix, task_labels,
        "COSINE SIMILARITY tra visual embedding (mean-pool, token_cat) — 10 task",
        fmt="{:.4f}",
    )
    print_matrix(
        euc_matrix, task_labels,
        "EUCLIDEAN DISTANCE tra visual embedding (mean-pool, token_cat) — 10 task",
        fmt="{:.2f}",
    )

    # ── Statistiche off-diagonal ──
    mask     = ~np.eye(n, dtype=bool)
    off_cos  = cos_matrix[mask]
    off_euc  = euc_matrix[mask]

    print(f"\n{'─'*80}")
    print(f"STATISTICHE (coppie off-diagonal, N={n*(n-1)})")
    print(f"  Cosine Similarity:   mean={off_cos.mean():.4f}  std={off_cos.std():.4f}  "
          f"min={off_cos.min():.4f}  max={off_cos.max():.4f}")
    print(f"  Euclidean Distance:  mean={off_euc.mean():.2f}   std={off_euc.std():.2f}   "
          f"min={off_euc.min():.2f}   max={off_euc.max():.2f}")

    # ── Interpretazione ──
    mean_cos = off_cos.mean()
    print(f"\n{'─'*80}")
    print("INTERPRETAZIONE:")
    if mean_cos > 0.99:
        print("  ● Cosine similarity molto alta (>0.99): i visual embedding sono quasi identici")
        print("    tra tutti i task. Il modello si affida PRINCIPALMENTE al prompt testuale.")
        print("    → Le differenze di performance su L1/L2/L3 sono attribuibili alla")
        print("      componente LINGUISTICA, non a quella visiva.")
    elif mean_cos > 0.95:
        print("  ● Cosine similarity alta (>0.95): il blocco visivo produce rappresentazioni")
        print("    simili ma non identiche. Il testo gioca comunque un ruolo dominante.")
    else:
        print("  ● Cosine similarity moderata: il blocco visivo differenzia i task.")
        print("    Sia la componente visiva che quella testuale contribuiscono.")

    # ── Sanity check ──
    task0          = task_suite.get_task(0)
    img0, wrist0   = get_first_frame(task0, 0, task_suite)
    sanity_check_text_independence(policy, image_processor, img0, wrist0)

    print(f"\n{'='*80}")
    print("Analisi completata.")


if __name__ == "__main__":
    main()
