import os

# هل نحن في كولاب؟
IS_COLAB = os.path.exists('/content')

if IS_COLAB:
    # إعدادات كولاب (للتدريب)
    BASE_DIR = "/content/drive/MyDrive/Face_Project"
    RAW_DATASET_DIR = os.path.join(BASE_DIR, "dataset", "raw")
    PROCESSED_DATASET_DIR = os.path.join(BASE_DIR, "dataset", "processed")
    TRAIN_DIR = os.path.join(PROCESSED_DATASET_DIR, "train")
    TEST_DIR = os.path.join(PROCESSED_DATASET_DIR, "test")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
else:
    # إعدادات السيرفر المحلي (للتشغيل)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATASET_DIR = None 
    PROCESSED_DATASET_DIR = None
    TRAIN_DIR = None
    TEST_DIR = None
    CHECKPOINT_DIR = None
    MODELS_DIR = BASE_DIR

# تأكد من وجود المجلدات في كولاب
if IS_COLAB and not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR, exist_ok=True)

# اسم النموذج ومساره
MODEL_FILENAME = "feature_extractor.h5"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

# الثوابت
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
