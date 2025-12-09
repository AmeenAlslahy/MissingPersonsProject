import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import config
from create_pairs import create_pairs
from build_model import build_siamese_model
import glob

# يعد دالة للعثور على أحدث Checkpoint
def get_latest_checkpoint(checkpoint_dir):
    list_of_files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.weights.h5"))
    if not list_of_files: return None, 0
    latest_file = max(list_of_files, key=os.path.getctime)
    try:
        filename = os.path.basename(latest_file)
        epoch_num = int(filename.split('_')[2].split('.')[0])
        return latest_file, epoch_num
    except: return None, 0

# دالة التدريب الرئيسية
def train():
    # 1. إعداد المجلدات
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)

    print("[INFO] Loading Data... جاري إنشاء الأزواج...")
    # إنشاء الأزواج
    pairs, labels = create_pairs(config.TRAIN_DIR)

    if len(pairs) == 0:
        print("Error: No pairs created. لم يتم انشاء أي أزواج. تحقق من بيانات التدريب.")
        return

    pairs = pairs.astype('float32')
    print(f"[INFO] Training on {len(pairs)} pairs... عدد الأزواج المستخدمة في التدريب.")

    # 2. إعداد مولد البيانات مع تحسينات
    datagen = ImageDataGenerator(
        rotation_range=25, # مقدار التدوير
        width_shift_range=0.15, # مقدار التحويل العرضي
        height_shift_range=0.15, # مقدار التحويل الطولي
        shear_range=0.15, # مقدار القص
        zoom_range=0.15, # مقدار التكبير/التصغير
        horizontal_flip=True, # عكس أفقي للصور
        brightness_range=[0.8, 1.2], # تعديل السطوع
        fill_mode='nearest', # وضع التعبئة
        preprocessing_function=preprocess_input # معالجة مسبقة لـ MobileNetV2
    )

    # 3. تعريف المولد
    def pair_generator():
        indices = np.arange(len(pairs))
        while True:
            np.random.shuffle(indices) # خلط البيانات كل عصر
            for i in range(0, len(pairs), config.BATCH_SIZE):
                batch_indices = indices[i:i+config.BATCH_SIZE]
                if len(batch_indices) == 0: continue

                batch_pairs = pairs[batch_indices]
                batch_labels = labels[batch_indices]

                imgA_batch = []
                imgB_batch = []

                for pair in batch_pairs:
                    imgA = datagen.random_transform(pair[0]) # تطبيق التحويلات العشوائية
                    imgA = datagen.standardize(imgA) # التقييس
                    imgB = datagen.random_transform(pair[1]) # تطبيق التحويلات العشوائية
                    imgB = datagen.standardize(imgB) # التقييس

                    imgA_batch.append(imgA)
                    imgB_batch.append(imgB)

                yield ((np.array(imgA_batch), np.array(imgB_batch)), np.array(batch_labels)) # إرجاع الدفعة

    # 4. تحويل المولد لـ Dataset
    train_dataset = tf.data.Dataset.from_generator(
        pair_generator,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )

    # 5. بناء النموذج
    print(f"[INFO] Building MobileNetV2 Model...")
    model, embedding_model = build_siamese_model(config.INPUT_SHAPE)

    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # 6. البحث عن Checkpoint
    latest_checkpoint, initial_epoch = get_latest_checkpoint(config.CHECKPOINT_DIR)
    if latest_checkpoint:
        print(f"[INFO] Resuming from epoch {initial_epoch}...")
        model.load_weights(latest_checkpoint)
    else:
        print("[INFO] Starting new training...")

    # 7. الكولباكس
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(config.CHECKPOINT_DIR, "model_epoch_{epoch:02d}.weights.h5"),
        save_weights_only=True,
        monitor='loss',
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)
    
    early_stop = EarlyStopping(
        monitor='loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # 8. بدء التدريب
    steps_per_epoch = max(1, len(pairs) // config.BATCH_SIZE)

    model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint, reduce_lr, early_stop]
    )

    print(f"[INFO] Saving Final Model...")
    embedding_model.save(config.MODEL_PATH)
    print(f"[INFO] Model saved to {config.MODEL_PATH}")

if __name__ == "__main__":
    train()