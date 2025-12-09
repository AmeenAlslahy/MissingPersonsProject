import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from IPython.display import Image, display
import config
from build_model import build_siamese_model

def find_target_layer(model):
    """
    البحث عن الطبقة التفافية المناسبة للنموذج السيامي - نسخة محسنة
    """
    print(f"[DEBUG] Searching for convolutional layers in model: {model.name}")
    
    # الحصول على الـ embedding network
    target_model = model
    
    # قائمة الطبقات المحتملة
    potential_layers = []
    
    # 1. البحث عن الطبقات التفافية في MobileNetV2
    for i, layer in enumerate(target_model.layers):
        layer_name = layer.name.lower()
        
        # طبقات MobileNetV2 الشائعة
        if any(keyword in layer_name for keyword in ['conv', 'block', 'depthwise', 'expand']):
            try:
                if hasattr(layer, 'output_shape'):
                    shape = layer.output_shape
                    # التحقق من أن الشكل رباعي الأبعاد (Batch, H, W, Channels)
                    if len(shape) == 4:
                        potential_layers.append(layer)
                        print(f"[DEBUG] Found potential layer {i}: {layer.name} - shape: {shape}")
            except:
                continue
    
    # 2. إذا لم نجد، نبحث في النموذج بأكمله
    if not potential_layers:
        for i, layer in enumerate(target_model.layers):
            try:
                if hasattr(layer, 'output_shape'):
                    shape = layer.output_shape
                    if len(shape) == 4:
                        potential_layers.append(layer)
                        print(f"[DEBUG] Found 4D layer {i}: {layer.name} - shape: {shape}")
            except:
                continue
    
    # 3. اختيار أفضل طبقة
    if not potential_layers:
        print("[DEBUG] Model layers summary:")
        for i, layer in enumerate(target_model.layers):
            print(f"  {i}: {layer.name} - {layer.__class__.__name__}")
            try:
                if hasattr(layer, 'output_shape'):
                    print(f"      shape: {layer.output_shape}")
            except:
                pass
        
        # محاولة استخدام طبقات محددة معروفة في MobileNetV2
        known_layers = [
            'block_16_project_BN',  # آخر طبقة في MobileNetV2
            'block_15_add',         # قبل الأخيرة
            'out_relu',             # إخراج MobileNetV2
            'Conv_1',               # طبقة الإدخال
            'global_average_pooling2d'  # قد تكون هذه موجودة
        ]
        
        for layer_name in known_layers:
            try:
                layer = target_model.get_layer(layer_name)
                potential_layers.append(layer)
                print(f"[DEBUG] Found known layer: {layer_name}")
                break
            except:
                continue
    
    if not potential_layers:
        # إذا لم نجد أي طبقة، نستخدم آخر طبقة قبل الـ GlobalAveragePooling
        for i, layer in enumerate(target_model.layers):
            if 'global_average_pooling' not in layer.name.lower():
                last_non_pool_layer = layer
        
        if last_non_pool_layer:
            potential_layers.append(last_non_pool_layer)
            print(f"[DEBUG] Using last non-pooling layer: {last_non_pool_layer.name}")
    
    if potential_layers:
        # اختيار الطبقة من منتصف الشبكة
        idx = len(potential_layers) // 2
        selected_layer = potential_layers[idx]
        print(f"[INFO] Selected layer: {selected_layer.name} (index {idx} of {len(potential_layers)})")
        return selected_layer.name
    else:
        # إذا لم نجد، نستخدم آخر طبقة
        last_layer = target_model.layers[-1]
        print(f"[WARNING] Using last layer as fallback: {last_layer.name}")
        return last_layer.name

def make_siamese_heatmap(img_tensor, model, target_layer_name, reference_tensor=None):
    """
    حساب خريطة التركيز للنموذج السيامي
    """
    try:
        # الحصول على الـ embedding network
        try:
            embedding_network = model.get_layer('Embedding_Network')
        except:
            embedding_network = model
        
        # إنشاء نموذج فرعي
        try:
            target_layer = embedding_network.get_layer(target_layer_name)
        except:
            print(f"[WARNING] Layer {target_layer_name} not found, using last layer")
            target_layer = embedding_network.layers[-1]
        
        grad_model = Model(
            inputs=embedding_network.input,
            outputs=[target_layer.output, embedding_network.output]
        )
        
        # استخدام صورة مرجعية إذا لم يتم توفيرها
        if reference_tensor is None:
            reference_tensor = img_tensor
        
        # حساب التدرجات
        with tf.GradientTape() as tape:
            # تمرير الصور
            conv_output1, embedding1 = grad_model(img_tensor)
            conv_output2, embedding2 = grad_model(reference_tensor)
            
            # حساب التشابه (Cosine similarity)
            similarity = tf.reduce_sum(embedding1 * embedding2, axis=-1)
            target = tf.reduce_mean(similarity)
        
        # حساب التدرجات
        grads = tape.gradient(target, conv_output1)
        
        if grads is None:
            print("[WARNING] Gradients are None, returning empty heatmap")
            # إرجاع خريطة بحجم صغير افتراضي
            return np.zeros((14, 14))
        
        # حساب الأهمية النسبية
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output1[0]
        
        # التأكد من الأبعاد
        if len(conv_output.shape) == 3 and len(pooled_grads.shape) == 1:
            heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
        else:
            # نسخة احتياطية
            heatmap = tf.reduce_mean(conv_output, axis=-1)
        
        # تطبيع
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        heatmap_np = heatmap.numpy()
        
        # تسجيل معلومات عن الخريطة
        print(f"[DEBUG] Heatmap shape: {heatmap_np.shape}, min: {heatmap_np.min():.4f}, max: {heatmap_np.max():.4f}")
        
        return heatmap_np
        
    except Exception as e:
        print(f"[ERROR] Heatmap generation failed: {e}")
        # إرجاع خريطة افتراضية
        return np.zeros((14, 14))

def prepare_image(img_path):
    """تحضير صورة للإدخال في النموذج"""
    try:
        img = image.load_img(img_path, target_size=config.INPUT_SHAPE[:2])
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return tf.convert_to_tensor(img_array, dtype=tf.float32)
    except Exception as e:
        print(f"[ERROR] Failed to prepare image {img_path}: {e}")
        return None

def debug_model_structure():
    """تصحيح هيكل النموذج"""
    print("\n" + "="*60)
    print("تصحيح هيكل النموذج")
    print("="*60)
    
    # بناء النموذج
    siamese_model, embedding_network = build_siamese_model(config.INPUT_SHAPE)
    
    # تحميل الأوزان
    if os.path.exists(config.MODEL_PATH):
        embedding_network.load_weights(config.MODEL_PATH)
        print("[INFO] Model weights loaded")
    
    # طباعة معلومات عن النموذج
    print("\n[INFO] Embedding Network Structure:")
    print(f"Model name: {embedding_network.name}")
    print(f"Number of layers: {len(embedding_network.layers)}")
    
    print("\n[INFO] First 10 layers:")
    for i, layer in enumerate(embedding_network.layers[:10]):
        print(f"  {i}: {layer.name} - {layer.__class__.__name__}")
        try:
            if hasattr(layer, 'output_shape'):
                print(f"      Output shape: {layer.output_shape}")
        except:
            pass
    
    print("\n[INFO] Last 10 layers:")
    for i, layer in enumerate(embedding_network.layers[-10:]):
        idx = len(embedding_network.layers) - 10 + i
        print(f"  {idx}: {layer.name} - {layer.__class__.__name__}")
        try:
            if hasattr(layer, 'output_shape'):
                print(f"      Output shape: {layer.output_shape}")
        except:
            pass
    
    print("\n[INFO] Searching for convolutional layers...")
    conv_layers = []
    for i, layer in enumerate(embedding_network.layers):
        try:
            if hasattr(layer, 'output_shape'):
                shape = layer.output_shape
                if len(shape) == 4:  # 4D layers (convolutional)
                    conv_layers.append(layer)
                    print(f"  ✓ {i}: {layer.name} - shape: {shape}")
        except:
            continue
    
    print(f"\n[INFO] Found {len(conv_layers)} convolutional layers")
    
    if conv_layers:
        # اختيار طبقة للاختبار
        test_layer = conv_layers[len(conv_layers)//2]
        print(f"\n[INFO] Recommending layer: {test_layer.name}")
        return test_layer.name
    else:
        print("\n[ERROR] No convolutional layers found!")
        return None

def analyze_ameen_images():
    """تحليل صورتين للشخص 'ameen' - نسخة مبسطة"""
    print("=" * 60)
    print("تحليل صورتين للشخص: ameen")
    print("=" * 60)
    
    test_dir = config.TEST_DIR
    ameen_dir = os.path.join(test_dir, "ameen")
    
    if not os.path.exists(ameen_dir):
        print("[ERROR] Directory 'ameen' not found in test folder")
        return
    
    # الحصول على جميع صور ameen
    ameen_images = []
    for file in os.listdir(ameen_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            ameen_images.append(os.path.join(ameen_dir, file))
    
    if len(ameen_images) < 2:
        print(f"[ERROR] Need at least 2 images for ameen, found {len(ameen_images)}")
        return
    
    print(f"[INFO] Found {len(ameen_images)} images for ameen")
    
    # استخدام أول صورتين
    img1_path = ameen_images[0]
    img2_path = ameen_images[1]
    
    print(f"[INFO] Image 1: {os.path.basename(img1_path)}")
    print(f"[INFO] Image 2: {os.path.basename(img2_path)}")
    
    # تحضير الصور
    print("\n[INFO] تحضير الصور للنموذج...")
    img1_tensor = prepare_image(img1_path)
    img2_tensor = prepare_image(img2_path)
    
    if img1_tensor is None or img2_tensor is None:
        print("[ERROR] Failed to prepare images")
        return
    
    # تحميل النموذج
    print("[INFO] تحميل النموذج السيامي...")
    siamese_model, embedding_network = build_siamese_model(config.INPUT_SHAPE)
    
    if os.path.exists(config.MODEL_PATH):
        embedding_network.load_weights(config.MODEL_PATH)
        print("[INFO] تم تحميل أوزان النموذج")
    else:
        print("[ERROR] ملف النموذج غير موجود")
        return
    
    try:
        # الحصول على الطبقة الموصى بها من دالة التصحيح
        print("[INFO] الحصول على الطبقة المناسبة...")
        recommended_layer = debug_model_structure()
        
        if recommended_layer:
            target_layer = recommended_layer
        else:
            # استخدام آخر طبقة كخيار احتياطي
            target_layer = embedding_network.layers[-1].name
            print(f"[WARNING] Using last layer: {target_layer}")
        
        # حساب التشابه مباشرة
        print("[INFO] حساب التشابه بين الصورتين...")
        emb1 = embedding_network.predict(img1_tensor.numpy(), verbose=0)[0]
        emb2 = embedding_network.predict(img2_tensor.numpy(), verbose=0)[0]
        similarity = np.dot(emb1, emb2)
        
        print(f"[RESULT] درجة التشابه: {similarity:.4f}")
        
        # محاولة حساب الخريطة الحرارية
        try:
            print("[INFO] محاولة حساب خريطة التركيز...")
            heatmap1 = make_siamese_heatmap(img1_tensor, siamese_model, target_layer, img2_tensor)
            heatmap2 = make_siamese_heatmap(img2_tensor, siamese_model, target_layer, img1_tensor)
            heatmaps_available = True
        except Exception as e:
            print(f"[WARNING] Cannot generate heatmaps: {e}")
            heatmaps_available = False
            heatmap1 = np.zeros((14, 14))
            heatmap2 = np.zeros((14, 14))
        
        # تحضير الصور للعرض
        img1_display = cv2.imread(img1_path)
        img1_display = cv2.resize(img1_display, (224, 224))
        img1_display_rgb = cv2.cvtColor(img1_display, cv2.COLOR_BGR2RGB)
        
        img2_display = cv2.imread(img2_path)
        img2_display = cv2.resize(img2_display, (224, 224))
        img2_display_rgb = cv2.cvtColor(img2_display, cv2.COLOR_BGR2RGB)
        
        # تفسير درجة التشابه
        if similarity > 0.7:
            similarity_text = "✅ تشابه عالي جداً (نفس الشخص بالتأكيد)"
            color = 'green'
        elif similarity > 0.5:
            similarity_text = "👍 تشابه جيد (نفس الشخص على الأرجح)"
            color = 'blue'
        elif similarity > 0.3:
            similarity_text = "⚠️ تشابه متوسط (قد يكون نفس الشخص)"
            color = 'orange'
        else:
            similarity_text = "❌ تشابه ضعيف (شخصين مختلفين)"
            color = 'red'
        
        # إنشاء التقرير النهائي
        fig = plt.figure(figsize=(15, 8))
        
        # الشبكة الرئيسية
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])
        
        # الصورة 1
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img1_display_rgb)
        ax1.set_title(f"الصورة الأولى\n{os.path.basename(img1_path)}", fontsize=12)
        ax1.axis('off')
        
        # الصورة 2
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img2_display_rgb)
        ax2.set_title(f"الصورة الثانية\n{os.path.basename(img2_path)}", fontsize=12)
        ax2.axis('off')
        
        # الخرائط الحرارية إذا كانت متاحة
        if heatmaps_available:
            # خريطة الصورة 1
            ax3 = fig.add_subplot(gs[0, 2])
            im3 = ax3.imshow(img1_display_rgb)
            overlay = ax3.imshow(heatmap1, cmap='jet', alpha=0.5)
            ax3.set_title("تركيز النموذج - الصورة 1", fontsize=12)
            ax3.axis('off')
            
            # خريطة الصورة 2 في صف جديد
            ax4 = fig.add_subplot(gs[1, 0])
            im4 = ax4.imshow(img2_display_rgb)
            ax4.imshow(heatmap2, cmap='jet', alpha=0.5)
            ax4.set_title("تركيز النموذج - الصورة 2", fontsize=12)
            ax4.axis('off')
            
            # شريط الألوان
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
            plt.colorbar(overlay, cax=cbar_ax)
            cbar_ax.set_ylabel('أهمية الميزة', rotation=270, labelpad=15)
        
        # مربع النتائج
        results_ax = fig.add_subplot(gs[1, 1:])
        results_ax.axis('off')
        
        results_text = (
            f"📊 نتائج التحليل:\n\n"
            f"🎯 درجة التشابه: {similarity:.4f}\n"
            f"📋 التقييم: {similarity_text}\n\n"
            f"🔍 تفسير النتائج:\n"
            f"• التشابه > 0.7: نفس الشخص بالتأكيد\n"
            f"• التشابه 0.5-0.7: نفس الشخص على الأرجح\n"
            f"• التشابه 0.3-0.5: يحتاج مزيداً من الفحص\n"
            f"• التشابه < 0.3: شخصين مختلفين\n\n"
            f"💡 معلومات تقنية:\n"
            f"• البعد التشابهي: {emb1.shape[0]}\n"
            f"• معيار المتجه 1: {np.linalg.norm(emb1):.4f}\n"
            f"• معيار المتجه 2: {np.linalg.norm(emb2):.4f}"
        )
        
        results_ax.text(0.05, 0.95, results_text, 
                       fontsize=11, 
                       verticalalignment='top',
                       color=color if color != 'green' else 'darkgreen',
                       transform=results_ax.transAxes)
        
        plt.suptitle(f"تحليل التعرف على الوجه - ameen", fontsize=16, y=0.98)
        plt.tight_layout()
        
        # حفظ النتيجة
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "ameen_face_analysis.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        
        print(f"\n✅ [SUCCESS] تم حفظ التحليل في: {output_path}")
        
        # عرض النتيجة
        print("\n" + "="*60)
        print("نتائج التحليل:")
        print("="*60)
        display(Image(filename=output_path))
        
        # معلومات إضافية للمطور
        print("\n📈 معلومات إضافية:")
        print(f"   - درجة التشابه: {similarity:.4f}")
        print(f"   - الزاوية بين المتجهين: {np.degrees(np.arccos(np.clip(similarity, -1, 1))):.1f}°")
        print(f"   - المسافة الإقليدية: {np.linalg.norm(emb1 - emb2):.4f}")
        
        if similarity > 0.5:
            print("\n🎉 الخلاصة: الصورتان لنفس الشخص (ameen)")
        else:
            print("\n⚠️ الخلاصة: قد تكون الصورتان لشخصين مختلفين")
            
    except Exception as e:
        print(f"[ERROR] فشل في التحليل: {e}")
        import traceback
        traceback.print_exc()

def main():
    """الوظيفة الرئيسية"""
    print("=" * 60)
    print("نظام تفسير قرارات النموذج السيامي")
    print("=" * 60)
    
    print("\nاختر نوع التحليل:")
    print("1. تحليل صورتين للشخص 'ameen'")
    print("2. تصحيح هيكل النموذج (للمطورين)")
    print("3. الخروج")
    
    try:
        choice = input("\nالرجاء إدخال رقم الخيار (1-3): ").strip()
        
        if choice == "1":
            analyze_ameen_images()
        elif choice == "2":
            debug_model_structure()
        elif choice == "3":
            print("[INFO] الخروج...")
        else:
            print("[ERROR] خيار غير صحيح")
            
    except KeyboardInterrupt:
        print("\n[INFO] العملية ألغيت")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
