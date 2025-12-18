from flask import Flask, render_template, request, send_from_directory, jsonify
import cv2
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# تنظيف الملفات القديمة كل ساعة
CLEANUP_INTERVAL = 3600
last_cleanup = time.time()

def cleanup_old_files():
    global last_cleanup
    now = time.time()
    if now - last_cleanup > CLEANUP_INTERVAL:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path) and now - os.path.getctime(file_path) > 86400:  # أقدم من 24 ساعة
                try:
                    os.remove(file_path)
                except:
                    pass
        last_cleanup = now

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    cleanup_old_files()

    # رفع صورة جديدة
    if 'image' in request.files:
        file = request.files['image']
        filename = secure_filename(file.filename)
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({'error': 'صيغة الملف غير مدعومة!'}), 400

        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'فشل في قراءة الصورة!'}), 400

        unique_id = str(uuid.uuid4())
        original_path = f"original_{unique_id}.jpg"
        original_fullpath = os.path.join(app.config['UPLOAD_FOLDER'], original_path)
        cv2.imwrite(original_fullpath, img)

        height, width, channels = img.shape
        file_size_kb = round(os.path.getsize(original_fullpath) / 1024, 2)
        img_format = filename.split('.')[-1].upper()

        properties = {
            'width': width,
            'height': height,
            'channels': channels,
            'size_kb': file_size_kb,
            'format': img_format
        }

        return jsonify({
            'original_image': f"uploads/{original_path}",
            'properties': properties
        })

    # معالجة صورة موجودة
    elif request.is_json:
        data = request.get_json()
        operation = data.get('operation')
        params = data.get('params', {})
        original_image = data.get('original_image')
        if not all([operation, original_image]):
            return jsonify({'error': 'بيانات ناقصة!'}), 400

        original_path = original_image.split('/')[-1]
        original_fullpath = os.path.join(app.config['UPLOAD_FOLDER'], original_path)
        img = cv2.imread(original_fullpath)
        if img is None:
            return jsonify({'error': 'الصورة الأصلية غير موجودة!'}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if operation == "grayscale":
            result = gray
        elif operation == "hist":
            result = cv2.equalizeHist(gray)
        elif operation == "blur":
            k = int(params.get('kernel_size', 15))
            k = k if k % 2 == 1 else k + 1
            k = max(1, k)
            result = cv2.GaussianBlur(gray, (k, k), 0)
        elif operation == "canny":
            low = int(params.get('low_threshold', 100))
            high = int(params.get('high_threshold', 200))
            result = cv2.Canny(img, low, high)
        elif operation == "sharpen":
            intensity = params.get('intensity', 1.0)
            kernel = np.array([[-1,-1,-1], [-1,9 + intensity,-1], [-1,-1,-1]])
            result = cv2.filter2D(gray, -1, kernel)
        elif operation == "threshold":
            thresh = int(params.get('thresh_value', 127))
            _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        elif operation == "cartoon":
            d = int(params.get('bilateral_d', 9))
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(img, d, 300, 300)
            result = cv2.bitwise_and(color, color, mask=~edges)
        elif operation == "rotate":
            angle = params.get('angle', 90)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(img, M, (w, h))
        elif operation == "resize":
            scale = params.get('scale', 50) / 100.0
            result = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        elif operation == "invert":
            result = cv2.bitwise_not(img)
        else:
            result = gray

        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        unique_id = str(uuid.uuid4())
        result_path = f"result_{unique_id}.jpg"
        result_fullpath = os.path.join(app.config['UPLOAD_FOLDER'], result_path)
        cv2.imwrite(result_fullpath, result)

        return jsonify({'result_image': f"uploads/{result_path}"})

    return jsonify({'error': 'طلب غير صالح!'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)