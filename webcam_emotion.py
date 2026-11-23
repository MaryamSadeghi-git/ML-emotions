import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ۱. لود کردن مدل آموزش‌دیده
MODEL_PATH = "raf_vgg16.keras"
print("در حال بارگذاری مدل...")
model = load_model(MODEL_PATH)
print("مدل با موفقیت لود شد.")

# ۲. تعریف برچسب‌ها دقیقاً طبق مپینگی که در کد آموزش داشتید:
# 0: عصبانی (Angry)
# 1: خوشحال (Happy)
# 2: غمگین (Sad)
# 3: خنثی (Neutral)
CLASS_NAMES = ["Angry", "Happy", "Sad", "Neutral"]

# ۳. آماده‌سازی آشکارساز چهره پیش‌فرض OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ۴. باز کردن وبکم (اگر دوربین باز نشد، 0 را به 1 تغییر دهید)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("خطا: وبکم باز نشد! اتصال دوربین را بررسی کنید.")
    exit()

print("\nوبکم فعال شد! برای خروج از برنامه، روی پنجره ویدیو کلیک کرده و کلید 'q' را فشار دهید.\n")

TARGET_SIZE = (100, 100)

while True:
    ret, frame = cap.read()
    if not ret:
        print("دریافت فریم از دوربین ناموفق بود.")
        break

    # تبدیل به تصویر خاکستری برای تشخیص دقیق‌تر موقعیت چهره
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(40, 40)
    )

    # اگر چهره‌ای در کادر نبود:
    if len(faces) == 0:
        cv2.putText(
            frame, 
            "No face detected!", 
            (30, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            (0, 0, 255), 
            2
        )
    else:
        # اگر چند نفر در تصویر بودند، بزرگ‌ترین چهره انتخاب می‌شود
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

        # بریدن بخش چهره از تصویر اصلی
        face_img = frame[y:y+h, x:x+w]

        # پیش‌پردازش دقیقاً مشابه فرآیند آموزش (تبدیل BGR به RGB و تغییر ابعاد به 100x100)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, TARGET_SIZE)
        
        # نرمال‌سازی بین 0 و 1 و افزودن بعد Batch: شکل خروجی (1, 100, 100, 3)
        face_input = np.array(face_img, dtype="float32") / 255.0
        face_input = np.expand_dims(face_input, axis=0)

        # پیش‌بینی با مدل
        preds = model.predict(face_input, verbose=0)[0]
        class_idx = np.argmax(preds)
        emotion = CLASS_NAMES[class_idx]
        confidence = preds[class_idx] * 100

        # رسم کادر دور چهره و نوشتن حالت تشخیص‌داده‌شده
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label_text = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(
            frame, 
            label_text, 
            (x, max(y - 10, 30)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2
        )

    # نمایش تصویر وبکم در یک پنجره
    cv2.imshow("Face Emotion Recognition - Press 'q' to Exit", frame)

    # بررسی فشردن کلید q برای خروج
    if cv2.waitKey(2000) & 0xFF == ord('q'):
        break

# آزادسازی منابع دوربین و بستن پنجره‌ها
cap.release()
cv2.destroyAllWindows()
