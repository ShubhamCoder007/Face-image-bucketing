import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils import cfg, logger

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# initialize once
_mtcnn = MTCNN(keep_all=True, device=device)
_embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def detect_faces(img_rgb):
    """Return list of bounding boxes or []"""
    boxes, _ = _mtcnn.detect(img_rgb)
    return [] if boxes is None else boxes

def embed_func(face_img):
    """
    Input: face_img as a H×W×3 RGB numpy array
    Returns: 1D torch.Tensor or None
    """
    if face_img is None or face_img.size == 0:
        logger.warning("embed_func: got empty crop")
        return None

    try:
        # ensure correct color & size
        face = cv2.resize(face_img, (160, 160))
        face_tensor = (
            torch.from_numpy(face)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
            / 255.0
        )
        with torch.no_grad():
            emb = _embedder(face_tensor).squeeze(0).cpu()
        if emb.numel() == 0:
            logger.warning("embed_func: got zero-length embedding")
            return None
        return emb
    except Exception as e:
        logger.warning(f"embed_func exception: {e}")
        return None
