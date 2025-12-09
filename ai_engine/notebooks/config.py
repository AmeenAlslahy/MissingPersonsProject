import os
# إعداد المسارات الأساسية للمشروع
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# مسار البيانات الخام
RAW_DATASET_DIR = os.path.join(BASE_DIR, "dataset", "raw")
# مسار البيانات المعالجة
PROCESSED_DATASET_DIR = os.path.join(BASE_DIR, "dataset", "processed")
# مسارات بيانات التدريب 
TRAIN_DIR = os.path.join(PROCESSED_DATASET_DIR, "train")
# مسارات بيانات الاختبار
TEST_DIR = os.path.join(PROCESSED_DATASET_DIR, "test")
# مسار حفظ النماذج
MODELS_DIR = os.path.join(BASE_DIR, "models")
# مسار حفظ النموذج النهائي
MODEL_PATH = os.path.join(MODELS_DIR, "feature_extractor.h5")

for directory in [MODELS_DIR, TRAIN_DIR, TEST_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# إعدادات جديدة لـ MobileNetV2
# حجم الصورة المدخلة
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32          # زدنا الحجم لاستقرار التدريب
# معدل التعلم
LEARNING_RATE = 0.0001   # قللنا المعدل لتدريب أدق
# عدد العصور (Epochs)
EPOCHS = 50

# إعدادات جوجل درايف
DRIVE_PATH = "/content/drive/MyDrive/Face_Project_Checkpoints"
CHECKPOINT_DIR = os.path.join(DRIVE_PATH, "checkpoints")