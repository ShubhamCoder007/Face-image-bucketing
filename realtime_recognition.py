import cv2
from utils import cfg, logger
from database_manager import load_database
from detect_and_embed import detect_faces, embed_func
from matcher import match_face

def main():
    # Load registered embeddings
    db = load_database()
    logger.info(f"Loaded identities: {list(db.keys())}")

    cap = cv2.VideoCapture(cfg['camera_index'])
    if not cap.isOpened():
        logger.error(f"Cannot open camera index {cfg['camera_index']}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            break

        # Convert to RGB for detection/embedding
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = detect_faces(rgb)

        # If no faces detected, just show the frame and continue
        if boxes is None or len(boxes) == 0:
            cv2.imshow("Live Recognition — press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Otherwise, process each detected face
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop and embed
            face = rgb[y1:y2, x1:x2]
            emb = embed_func(face)

            # Match to a registered identity (or 'Unknown')
            name = match_face(emb, db) if emb is not None else 'Unknown'

            # Draw the name above the box
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("Live Recognition — press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
