import os
import pickle
import torch
from utils import cfg, logger

def load_database():
    db = {}
    for fname in os.listdir(cfg['paths']['embeddings_dir']):
        if not fname.endswith('.pkl'):
            continue
        person = fname[:-4]
        with open(os.path.join(cfg['paths']['embeddings_dir'], fname),'rb') as f:
            embs = pickle.load(f)
        # filter None
        valid = [e for e in embs if isinstance(e, torch.Tensor)]
        if not valid:
            logger.warning(f"No valid embeddings for {person}; skipping")
            continue
        all_tensor = torch.stack(valid)
        db[person] = all_tensor
    return db
