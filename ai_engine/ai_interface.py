
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
try:
    from . import config
except ImportError:
    import config

class FaceNet:
    _instance = None
    model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceNet, cls).__new__(cls)
            cls._instance.load_extractor()
        return cls._instance

    def load_extractor(self):
        print(f"[AI] Loading model from: {config.MODEL_PATH}")
        if os.path.exists(config.MODEL_PATH):
            try:
                self.model = load_model(config.MODEL_PATH, compile=False)
                print("[AI] Model loaded ✅")
            except Exception as e:
                print(f"[AI] Error: {e} ❌")
        else:
            print(f"[AI] Warning: No model found at {config.MODEL_PATH}")

    def get_embedding(self, img_path):
        if not self.model: return None
        try:
            img = image.load_img(img_path, target_size=config.INPUT_SHAPE[:2])
            img_arr = image.img_to_array(img)
            img_arr = np.expand_dims(img_arr, axis=0)
            img_arr = preprocess_input(img_arr)
            emb = self.model.predict(img_arr, verbose=0)[0]
            return emb.tolist()
        except Exception as e:
            print(f"[AI] Process Error: {e}")
            return None
            
    @staticmethod
    def compute_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
