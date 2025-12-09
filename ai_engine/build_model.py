import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.backend as K
import config

def cosine_distance(vectors):
    """
    حساب تشابه جيب التمام (Cosine Similarity)
    بما أن المتجهات مطبعة (Normalized)، فهو يساوي Dot Product
    """
    (featsA, featsB) = vectors
    return K.sum(featsA * featsB, axis=1, keepdims=True)

def build_siamese_model(input_shape):
    print("[INFO] Building Model with MobileNetV2 Backbone...")
    
    # 1. القاعدة: MobileNetV2
    base_cnn = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    
    # تجميد الطبقات الأولى للحفاظ على الميزات الأساسية
    for layer in base_cnn.layers[:-30]:
        layer.trainable = False
    for layer in base_cnn.layers[-30:]:
        layer.trainable = True

    # 2. الرأس (Head) - مبسط لتقليل الـ Overfitting
    x = base_cnn.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    
    # التطبيع (Normalization) - جوهري لـ Cosine Similarity
    x = Lambda(lambda v: K.l2_normalize(v, axis=1))(x)

    embedding_network = Model(base_cnn.input, x, name="Embedding_Network")

    # 3. الشبكة التوأمية
    imgA = Input(shape=input_shape)
    imgB = Input(shape=input_shape)

    featsA = embedding_network(imgA)
    featsB = embedding_network(imgB)

    # حساب المسافة (Cosine)
    distance = Lambda(cosine_distance)([featsA, featsB])
    
    # طبقة القرار (Sigmoid)
    outputs = Dense(1, activation="sigmoid")(distance)

    model = Model(inputs=[imgA, imgB], outputs=outputs)
    
    return model, embedding_network