import torch
from torch.nn.functional import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from utils import cfg, logger

# optional KNN model cache
_knn = None

def match_face(emb, database):
    if not database:
        return 'Unknown'

    # brute-force cosine
    best, score = 'Unknown', -1.0
    for person, embs in database.items():
        sims = cosine_similarity(emb.unsqueeze(0), embs)
        m = sims.max().item()
        if m > score:
            score, best = m, person
    if score>=cfg['matching']['threshold']:
        return best
    return 'Unknown'
