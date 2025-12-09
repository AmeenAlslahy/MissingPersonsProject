import os
import numpy as np
import cv2
import random
import config

def create_pairs(dataset_dir):
    print(f"[INFO] Creating Balanced Pairs from: {dataset_dir}")
    
    person_images_map = {}
    if not os.path.exists(dataset_dir):
        return np.array([]), np.array([])

    all_people = os.listdir(dataset_dir)
    for person_name in all_people:
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir): continue
        
        images = []
        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, config.INPUT_SHAPE[:2])
                images.append(img)
        
        if len(images) > 0:
            person_images_map[person_name] = images

    people_names = list(person_images_map.keys())
    pairs = []
    labels = []

    for person_name, person_imgs in person_images_map.items():
        for i in range(len(person_imgs)):
            # --- زوج موجب (Positive) ---
            if len(person_imgs) > 1:
                next_idx = (i + 1) % len(person_imgs)
                pairs.append([person_imgs[i], person_imgs[next_idx]])
                labels.append(1.0)

            # --- زوج سالب (Negative) ---
            # 1:1 للتوازن
            random_person = random.choice(people_names)
            while random_person == person_name:
                random_person = random.choice(people_names)
            
            if len(person_images_map[random_person]) > 0:
                random_img = random.choice(person_images_map[random_person])
                pairs.append([person_imgs[i], random_img])
                labels.append(0.0)

    # إحصائيات
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    print(f"[STATS] Positives: {pos_count}, Negatives: {neg_count}")
    print(f"[STATS] Positive Ratio: {pos_count/len(labels)*100:.1f}%")

    # خلط البيانات
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    if len(combined) > 0:
        pairs[:], labels[:] = zip(*combined)

    print(f"[INFO] Generated {len(pairs)} balanced pairs.")
    return np.array(pairs), np.array(labels)