# threshold_tuner.py

import itertools
import yaml
import numpy as np
import torch
from torch.nn.functional import cosine_similarity

from database_manager import load_database
from utils import cfg, logger

def compute_similarities(db):
    intra, inter = [], []
    persons = list(db.keys())

    # Intra-class: all pairs within each person
    for person in persons:
        embs = db[person]                # Tensor shape (N_p, D)
        n = embs.shape[0]
        if n < 2:
            continue
        for i, j in itertools.combinations(range(n), 2):
            sim = cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
            intra.append(sim)

    # Inter-class: all pairs across different persons
    for p1, p2 in itertools.combinations(persons, 2):
        e1s, e2s = db[p1], db[p2]
        for e1 in e1s:
            for e2 in e2s:
                sim = cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
                inter.append(sim)

    return np.array(intra), np.array(inter)

def find_equal_error_threshold(intra, inter, steps=101):
    best_t, best_diff = 0.0, float('inf')
    candidates = np.linspace(0, 1, steps)
    for t in candidates:
        # False Reject Rate = fraction of same-person sims BELOW threshold
        frr = np.mean(intra < t) if intra.size else 0.0
        # False Accept Rate  = fraction of different-person sims ABOVE threshold
        far = np.mean(inter >= t) if inter.size else 0.0
        diff = abs(frr - far)
        if diff < best_diff:
            best_diff, best_t = diff, t
    return best_t

def main():
    # 1) Load database
    db = load_database()
    if not db:
        logger.error("No embeddings found in database; aborting.")
        return
    logger.info(f"Loaded {len(db)} identities for threshold tuning")

    # 2) Compute similarity distributions
    intra, inter = compute_similarities(db)
    logger.info(f"Intra-class pairs: {len(intra)}, Inter-class pairs: {len(inter)}")

    # 3) Find threshold at Equal Error Rate
    thresh = find_equal_error_threshold(intra, inter, steps=201)
    logger.info(f"Recommended similarity threshold ≈ {thresh:.3f}")

    # 4) (Optional) Update config.yaml
    resp = input("Update config.yaml with this threshold? (y/N): ").strip().lower()
    if resp == 'y':
        cfg['matching']['threshold'] = float(thresh)
        with open('config.yaml', 'w') as f:
            yaml.safe_dump(cfg, f)
        logger.info("config.yaml updated.")
    else:
        logger.info("config.yaml not modified.")

if __name__ == "__main__":
    main()
