# capture_faces.py

import os
import cv2
import pickle
import yaml

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from utils import cfg, logger
from detect_and_embed import detect_faces, embed_func
from database_manager import load_database
from threshold_tuner import compute_similarities, find_equal_error_threshold

def main():
    # 1) Get person name and prepare output folder
    person = input("Enter person name (no spaces): ").strip()
    out_dir = cfg['paths']['embeddings_dir']
    os.makedirs(out_dir, exist_ok=True)

    # 2) Initialize webcam and counters
    cap = cv2.VideoCapture(cfg['camera_index'])
    if not cap.isOpened():
        logger.error(f"Cannot open camera index {cfg['camera_index']}")
        return

    embeddings = []
    count = 0
    burst_limit = cfg['capture']['burst_limit']

    # 3) Capture burst of faces
    while count < burst_limit:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from webcam")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = detect_faces(rgb)

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                face = rgb[y1:y2, x1:x2]
                emb  = embed_func(face)
                if emb is not None:
                    embeddings.append(emb)
                    count += 1
                else:
                    logger.warning("Skipped invalid embedding.")

                if count >= burst_limit:
                    break

        # Show capture progress
        cv2.putText(frame,
                    f"Captured: {count}/{burst_limit}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)
        cv2.imshow("Register Face – press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4) Cleanup camera
    cap.release()
    cv2.destroyAllWindows()

    logger.info(f"Total embeddings captured for {person}: {len(embeddings)}")
    if not embeddings:
        logger.error("No embeddings captured—check face detection!")
        return

    # 5) Save embeddings list
    save_path = os.path.join(out_dir, f"{person}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(embeddings, f)
    logger.info(f"Saved embeddings to {save_path}")

    # ----- AUTOMATIC THRESHOLD TUNING -----

    # 6) Reload full database (including new person)
    db = load_database()
    if not db:
        logger.error("No registered identities in database; skipping threshold tuning.")
        return

    # 7) Compute intra/inter similarities
    intra, inter = compute_similarities(db)
    logger.info(f"Computed {len(intra)} intra-class and {len(inter)} inter-class similarities")

    # 8) Find new Equal-Error-Rate threshold
    new_thresh = find_equal_error_threshold(intra, inter, steps=201)
    logger.info(f"Auto-tuned similarity threshold → {new_thresh:.3f}")

    # 9) Update config.yaml on disk
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config.setdefault('matching', {})['threshold'] = float(new_thresh)

    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)
    logger.info("config.yaml updated with new threshold")

if __name__ == "__main__":
    main()
