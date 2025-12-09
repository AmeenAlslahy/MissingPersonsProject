import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import config
import os

# تحميل النموذج مرة واحدة
print("[AI ENGINE] Loading Feature Extractor Model...")
try:
    if os.path.exists(config.MODEL_PATH):
        feature_model = load_model(config.MODEL_PATH)
        print("[AI ENGINE] Model loaded successfully.")
    else:
        print(f"[WARNING] Model file not found at {config.MODEL_PATH}")
        feature_model = None
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    feature_model = None

class FaceEngine:
    
    @staticmethod
    def preprocess_image(img_path):
        """
        تجهيز الصورة لتناسب MobileNetV2
        """
        # 1. قراءة الصورة
        img = cv2.imread(img_path)
        if img is None: return None
        
        # 2. تغيير الحجم
        img = cv2.resize(img, config.INPUT_SHAPE[:2])
        
        # 3. تحويل الألوان إلى RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 4. تحويل لمصفوفة وإضافة البعد الرابع
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 5. معالجة الألوان الخاصة بـ MobileNetV2
        img_array = preprocess_input(img_array)
        
        return img_array

    @staticmethod
    def get_embedding(img_path):
        """
        استخراج البصمة (Vector)
        """
        if feature_model is None:
            print("[ERROR] Model is not loaded.")
            return None

        processed_img = FaceEngine.preprocess_image(img_path)
        if processed_img is None: return None
        
        vector = feature_model.predict(processed_img, verbose=0)
        
        return vector[0]

    @staticmethod
    def calculate_similarity(vector_a, vector_b):
        """
        حساب التشابه (Cosine Similarity)
        """
        vec_a = np.array(vector_a)
        vec_b = np.array(vector_b)
        
        return np.dot(vec_a, vec_b)