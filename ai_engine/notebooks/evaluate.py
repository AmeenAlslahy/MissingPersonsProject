import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import random
import seaborn as sns
import config
from build_model import build_siamese_model

def get_embedding(model, img_path):
    """
    استخراج البصمة الرقمية (Vector) من الصورة.
    """
    try:
        img = load_img(img_path, target_size=config.INPUT_SHAPE[:2])
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # استخدام preprocess_input الخاص بـ MobileNetV2
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        embedding = model.predict(img_array, verbose=0)[0]
        return embedding
    except Exception as e:
        return None

def check_embeddings(model, person_imgs):
    """فحص جودة المتجهات"""
    if not person_imgs:
        return
    
    first_person = list(person_imgs.keys())[0]
    test_img = person_imgs[first_person][0]
    emb = get_embedding(model, test_img)
    
    print(f"\n[DEBUG] Embedding Analysis:")
    print(f"  Shape: {emb.shape}")
    print(f"  Norm: {np.linalg.norm(emb):.6f} (should be ≈1.0)")
    print(f"  Min/Max: {emb.min():.4f}, {emb.max():.4f}")
    
    # تحقق من متشابهات لنفس الشخص
    if len(person_imgs[first_person]) >= 2:
        emb1 = get_embedding(model, person_imgs[first_person][0])
        emb2 = get_embedding(model, person_imgs[first_person][1])
        sim = np.dot(emb1, emb2)
        print(f"  Same person similarity: {sim:.4f}")

def evaluate():
    print("[INFO] Building Model Architecture...")
    _, embedding_network = build_siamese_model(config.INPUT_SHAPE)
    
    print("[INFO] Loading Weights from:", config.MODEL_PATH)
    if not os.path.exists(config.MODEL_PATH):
        print("[ERROR] Weights file not found. Please train the model first.")
        return

    try:
        embedding_network.load_weights(config.MODEL_PATH)
        print("[SUCCESS] Weights loaded.")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        return

    model = embedding_network

    # --- تجهيز بيانات الاختبار ---
    print("[INFO] Generating Test Pairs...")
    if not os.path.exists(config.TEST_DIR):
        print("[ERROR] Test directory not found.")
        return

    people = os.listdir(config.TEST_DIR)
    person_imgs = {}
    
    for p in people:
        path = os.path.join(config.TEST_DIR, p)
        if os.path.isdir(path):
            imgs = [os.path.join(path, f) for f in os.listdir(path) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if len(imgs) > 0:
                person_imgs[p] = imgs

    all_people = list(person_imgs.keys())
    print(f"[DEBUG] Found {len(all_people)} people in Test Set.")
    
    if len(all_people) < 2:
        print("[WARNING] Not enough people for evaluation (Need at least 2).")
        return

    # فحص المتجهات
    check_embeddings(model, person_imgs)

    pairs = []
    labels = [] # 1: نفس الشخص (Similar), 0: مختلف (Different)

    # توليد الأزواج
    for p, imgs in person_imgs.items():
        # 1. أزواج موجبة (Positive Pairs)
        if len(imgs) > 1:
            for i in range(len(imgs)):
                for j in range(i+1, len(imgs)):
                    pairs.append((imgs[i], imgs[j]))
                    labels.append(1)
        
        # 2. أزواج سالبة (Negative Pairs)
        num_negatives = len(imgs)
        for _ in range(num_negatives):
            p2 = random.choice(all_people)
            while p2 == p:
                p2 = random.choice(all_people)
            
            if len(person_imgs[p2]) > 0:
                img1 = random.choice(imgs)
                img2 = random.choice(person_imgs[p2])
                pairs.append((img1, img2))
                labels.append(0)

    if len(pairs) == 0:
        print("[ERROR] No pairs generated.")
        return

    print(f"[INFO] Evaluating on {len(pairs)} pairs...")
    y_true, y_scores = [], []

    # التقييم الفعلي
    for i, (p1, p2) in enumerate(pairs):
        emb1 = get_embedding(model, p1)
        emb2 = get_embedding(model, p2)
        
        if emb1 is not None and emb2 is not None:
            # حساب التشابه باستخدام جيب التمام (Cosine Similarity)
            score = np.dot(emb1, emb2)
            
            y_true.append(labels[i])
            y_scores.append(score)
            
        if i > 0 and i % 100 == 0:
            print(f"   Processed {i}/{len(pairs)} pairs...")

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    if len(y_true) == 0:
        print("[ERROR] No valid scores obtained.")
        return

    # --- البحث عن أفضل عتبة ---
    best_acc = 0
    best_thresh = 0
    
    # التشابه يتراوح من -1 إلى 1
    for thresh in np.arange(-1.0, 1.0, 0.01):
        # إذا التشابه أكبر من العتبة، فهو تطابق (1)
        y_pred = (y_scores > thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    print(f"\n>>> EVALUATION RESULTS <<<")
    print(f"Optimal Similarity Threshold: {best_thresh:.4f}")
    print(f"Best Accuracy: {best_acc*100:.2f}%")
    
    # --- ROC Curve ---
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")

    # --- حفظ الرسوم البيانية ---
    output_dir = os.path.join(config.BASE_DIR, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Similarity Distribution
    pos_sims = y_scores[y_true == 1]
    neg_sims = y_scores[y_true == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(pos_sims, bins=30, alpha=0.6, label='Same Person (High Sim)', color='green')
    plt.hist(neg_sims, bins=30, alpha=0.6, label='Different Person (Low Sim)', color='red')
    plt.axvline(best_thresh, color='black', linestyle='dashed', linewidth=2, label=f'Threshold ({best_thresh:.2f})')
    
    plt.title(f'Cosine Similarity Distribution (Accuracy: {best_acc*100:.1f}%)')
    plt.xlabel('Similarity (-1.0 to 1.0)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'))
    
    # 2. Confusion Matrix
    final_pred = (y_scores > best_thresh).astype(int)
    cm = confusion_matrix(y_true, final_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Different', 'Same'], 
                yticklabels=['Different', 'Same'])
    plt.title(f'Confusion Matrix (Acc: {best_acc*100:.1f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # 3. ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))

    print(f"[INFO] All plots saved to: {output_dir}")

if __name__ == "__main__":
    evaluate()