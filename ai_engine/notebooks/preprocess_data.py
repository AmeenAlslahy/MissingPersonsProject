import cv2
import os
import mediapipe as mp
import config
import shutil
import random

def split_and_save(person_name, faces_list):
    """ تقسيم الصور وحفظها في Train/Test """
    random.shuffle(faces_list)
    total = len(faces_list)
    test_count = int(total * 0.2)
    
    if test_count == 0 and total > 1: test_count = 1
    elif total == 1: test_count = 0

    test_imgs = faces_list[:test_count]
    train_imgs = faces_list[test_count:]

    train_path = os.path.join(config.TRAIN_DIR, person_name)
    test_path = os.path.join(config.TEST_DIR, person_name)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for i, img in enumerate(train_imgs):
        cv2.imwrite(os.path.join(train_path, f"{i}.jpg"), img)
    for i, img in enumerate(test_imgs):
        cv2.imwrite(os.path.join(test_path, f"{i}.jpg"), img)

    return len(train_imgs), len(test_imgs)

def run_preprocessing():
    # تنظيف المجلدات القديمة
    if os.path.exists(config.PROCESSED_DATASET_DIR):
        shutil.rmtree(config.PROCESSED_DATASET_DIR)
    os.makedirs(config.TRAIN_DIR)
    os.makedirs(config.TEST_DIR)

    # إعداد MediaPipe
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7)

    print("[INFO] Starting Processing (MediaPipe + Split)...")
    total_train, total_test = 0, 0

    if not os.path.exists(config.RAW_DATASET_DIR):
        print("Error: Raw dataset not found.")
        return

    for person_name in os.listdir(config.RAW_DATASET_DIR):
        person_path = os.path.join(config.RAW_DATASET_DIR, person_name)
        if not os.path.isdir(person_path): continue

        print(f" -> Processing: {person_name}")
        valid_faces = []
        
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            try:
                image = cv2.imread(img_path)
                if image is None: continue
                
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb_image)
                
                if not results.detections: continue

                # أخذ أكبر وجه فقط
                h, w, c = image.shape
                largest = max(results.detections, 
                            key=lambda d: d.location_data.relative_bounding_box.width * 
                                         d.location_data.relative_bounding_box.height)
                bbox = largest.location_data.relative_bounding_box
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                fw = int(bbox.width * w)
                fh = int(bbox.height * h)
                
                pad_w = int(fw * 0.2)
                pad_h = int(fh * 0.2)
                
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(w, x + fw + pad_w)
                y2 = min(h, y + fh + pad_h)
                
                face = image[y1:y2, x1:x2]
                if face.size == 0: continue
                
                # تغيير الحجم للحفاظ على نسبة الأبعاد
                face_resized = cv2.resize(face, config.INPUT_SHAPE[:2])
                valid_faces.append(face_resized)

            except Exception as e:
                print(f"Error in {img_name}: {e}")

        if valid_faces:
            tr, te = split_and_save(person_name, valid_faces)
            total_train += tr
            total_test += te

    print(f"\n[INFO] Done. Train: {total_train}, Test: {total_test}")

if __name__ == "__main__":
    run_preprocessing()