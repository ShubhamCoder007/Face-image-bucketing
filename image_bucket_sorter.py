#!/usr/bin/env python3
import os
import shutil
import cv2
import argparse

from utils import cfg, logger
from database_manager import load_database
from detect_and_embed import detect_faces, embed_func
from matcher import match_face

def main():
    parser = argparse.ArgumentParser(
        description="Sort images into per-person folders based on face recognition"
    )
    parser.add_argument(
        "-i", "--input_dir", required=True,
        help="Folder containing images to sort"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True,
        help="Base folder where per-person subfolders will be created"
    )
    args = parser.parse_args()

    # Load registered embeddings
    db = load_database()
    if not db:
        logger.error("No registered identities found; aborting.")
        return
    logger.info(f"Registered identities: {list(db.keys())}")

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    max_dim = cfg.get('preproc', {}).get('max_dim', 800)

    # Process each image in the input directory
    for img_name in os.listdir(args.input_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        in_path = os.path.join(args.input_dir, img_name)

        # Fast, half-resolution read for JPEGs
        frame = cv2.imread(in_path, cv2.IMREAD_REDUCED_COLOR_2)
        if frame is None:
            logger.warning(f"Could not read {img_name}, skipping.")
            continue

        # Further downscale if larger than max_dim
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        # Convert to RGB for detection & embedding
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = detect_faces(rgb)
        if boxes is None or len(boxes) == 0:
            logger.info(f"No face detected in {img_name}, skipping.")
            continue

        # Match every detected face, collect unique identities
        identified = set()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = rgb[y1:y2, x1:x2]
            emb = embed_func(face)
            if emb is None:
                logger.warning(f"Embedding failed for {img_name}, skipping this face.")
                continue
            person = match_face(emb, db)
            if person != 'Unknown':
                identified.add(person)

        # Copy original file into each identified person's folder
        for person in identified:
            person_dir = os.path.join(args.output_dir, person)
            os.makedirs(person_dir, exist_ok=True)
            dst = os.path.join(person_dir, img_name)
            shutil.copy(in_path, dst)
            logger.info(f"{img_name} → {person}")

    logger.info("Image sorting complete.")

if __name__ == "__main__":
    main()
