# register_from_images.py

import os
import pickle
import argparse

import cv2
import torch
import numpy as np
import yaml

from utils import cfg, logger
from detect_and_embed import detect_faces, embed_func
from database_manager import load_database
from matcher import match_face  # just to be sure matcher is imported

# Import the threshold‐tuning functions
from threshold_tuner import compute_similarities, find_equal_error_threshold

def register_from_folder(person_name, images_dir):
    all_embs = []

    # 1) Embed every valid face
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(('.jpg','jpeg','png')):
            continue
        path = os.path.join(images_dir, fname)
        img = cv2.imread(path)
        if img is None:
            logger.warning(f"Cannot read {fname}, skipping.")
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = detect_faces(rgb)
        if boxes is None or len(boxes)==0:
            logger.warning(f"No face in {fname}, skipping.")
            continue

        x1,y1,x2,y2 = map(int, boxes[0])
        face = rgb[y1:y2, x1:x2]
        emb = embed_func(face)
        if emb is not None:
            all_embs.append(emb)
        else:
            logger.warning(f"Embedding failed for {fname}, skipping.")

    if not all_embs:
        logger.error("No valid embeddings found—aborting registration.")
        return

    # 2) Stack and compute mean
    embs = torch.stack(all_embs)         # shape (N, D)
    mean_emb = embs.mean(dim=0)          # (D,)

    # 3) Compute distances and pick top‐K outliers (so total ≤20)
    diffs = embs - mean_emb.unsqueeze(0)
    dists = torch.norm(diffs, dim=1).cpu().numpy()
    N = len(all_embs)
    K = min(N-1, cfg['capture']['max_outliers'])  # e.g. max_outliers=19 in config.yaml
    outlier_idxs = np.argsort(-dists)[:K]

    # 4) Build final list: mean + outliers
    final_embs = [ mean_emb ] + [ all_embs[i] for i in outlier_idxs ]

    # 5) Save that list of Tensors
    out_dir = cfg['paths']['embeddings_dir']
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{person_name}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(final_embs, f)

    logger.info(
        f"Registered {person_name}: 1 mean + {K} outliers "
        f"(total {1+K}) from {N} input images → saved to {save_path}"
    )

    # ----- AUTOMATIC THRESHOLD TUNING -----
    # Reload full database including this new person
    db = load_database()
    intra, inter = compute_similarities(db)
    new_thresh = find_equal_error_threshold(intra, inter, steps=201)
    logger.info(f"Tuned similarity threshold to {new_thresh:.3f}")

    # Update config.yaml on disk
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config.setdefault('matching', {})['threshold'] = float(new_thresh)

    with open('config.yaml', 'w') as f:
        yaml.safe_dump(config, f)

    logger.info("config.yaml updated with new threshold.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Register a person from a folder of images, then retune threshold"
    )
    p.add_argument("--person_name", "-n", required=True,
                   help="Identifier for this person (no spaces)")
    p.add_argument("--images_dir",  "-i", required=True,
                   help="Folder containing face images for this person")
    args = p.parse_args()

    register_from_folder(args.person_name, args.images_dir)
