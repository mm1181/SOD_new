"""
消防器材检测 - 纯 YOLO 独立运行入口

功能：
1. 独立运行，不依赖 SOD 模型
2. YOLO 同时检测消防器材和动火点
3. 提供 Web API 和图片上传检测

运行方式：
    python fire_app.py

API 端点：
    POST /detect - 上传图片进行检测
    GET / - 检测页面
"""

import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from PIL import Image

from fire_detection import FireEquipmentService

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/fire_uploads'
app.config['RESULT_FOLDER'] = 'static/fire_results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 创建必要的目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 全局服务实例
fire_service = None


def load_service():
    """加载消防检测服务"""
    global fire_service
    fire_service = FireEquipmentService(
        model_path="fire_detection/weights/fire_equipment.pt",
        cooldown_seconds=5
    )
    print("✅ 消防器材检测服务已初始化")


def process_image(image_path):
    """
    处理上传的图片
    
    Args:
        image_path: 图片路径
    
    Returns:
        dict: 检测结果
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "无法读取图片"}
    
    original_size = image.shape[:2]  # (H, W)
    
    # 执行检测
    is_alarm, alarm_details = fire_service.detect_fire_equipment(image, conf=0.5)
    
    # 获取原始检测结果用于绘图
    raw_result = fire_service.get_raw_detection(image, conf=0.5)
    
    # 绘制检测结果
    result_image = fire_service.draw_detection_result(image, raw_result)
    
    # 生成唯一文件名
    file_id = str(uuid.uuid4())[:8]
    
    # 保存结果图片
    result_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_result.jpg')
    cv2.imwrite(result_path, result_image)
    
    # 构建返回结果
    result = {
        "file_id": file_id,
        "result_image": f'fire_results/{file_id}_result.jpg',
        "is_alarm": is_alarm,
        "alarm_details": alarm_details,
        "detection": {
            "equipment_count": len(raw_result.get("equipment", [])) if raw_result else 0,
            "fire_zone_count": len(raw_result.get("fire_zones", [])) if raw_result else 0,
            "equipment": raw_result.get("equipment", []) if raw_result else [],
            "fire_zones": [
                {k: v for k, v in z.items() if k != "contour"}
                for z in raw_result.get("fire_zones", [])
            ] if raw_result else [],
        }
    }
    
    return result


@app.route('/')
def index():
    """检测页面"""
    return render_template('fire_index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """
    图片检测 API
    
    请求：multipart/form-data，字段名 'file'
    响应：JSON 格式检测结果
    """
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 检查文件格式
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({'error': f'不支持的文件格式: {ext}'}), 400
    
    # 保存上传的文件
    file_id = str(uuid.uuid4())[:8]
    filename = f'{file_id}{ext}'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # 处理图片
    result = process_image(filepath)
    result['original'] = f'fire_uploads/{filename}'
    
    return jsonify(result)


@app.route('/detect_base64', methods=['POST'])
def detect_base64():
    """
    Base64 图片检测 API
    
    请求：JSON，字段 'image' 为 base64 编码的图片
    响应：JSON 格式检测结果
    """
    import base64
    
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': '缺少 image 字段'}), 400
    
    try:
        # 解码 base64
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': '无法解码图片'}), 400
        
        # 保存临时文件
        file_id = str(uuid.uuid4())[:8]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.jpg')
        cv2.imwrite(filepath, image)
        
        # 处理图片
        result = process_image(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'service': 'fire_detection',
        'mode': 'yolo_only'
    })


@app.route('/static/<path:filename>')
def static_files(filename):
    """静态文件服务"""
    return send_from_directory('static', filename)


# 创建简单的前端页面
FIRE_INDEX_HTML = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>消防器材检测 - 纯YOLO模式</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 10px;
            font-size: 2em;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }
        .mode-badge {
            display: inline-block;
            background: #e74c3c;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        .upload-area {
            background: rgba(255,255,255,0.05);
            border: 2px dashed rgba(255,255,255,0.2);
            border-radius: 16px;
            padding: 60px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 30px;
        }
        .upload-area:hover {
            border-color: #3498db;
            background: rgba(52, 152, 219, 0.1);
        }
        .upload-area.dragover {
            border-color: #2ecc71;
            background: rgba(46, 204, 113, 0.1);
        }
        .upload-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        #fileInput { display: none; }
        .results {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .results { grid-template-columns: 1fr; }
        }
        .result-card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
        }
        .result-card h3 {
            margin-bottom: 15px;
            color: #3498db;
        }
        .result-card img {
            width: 100%;
            border-radius: 8px;
        }
        .alarm-box {
            background: rgba(231, 76, 60, 0.2);
            border: 1px solid #e74c3c;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }
        .alarm-box.safe {
            background: rgba(46, 204, 113, 0.2);
            border-color: #2ecc71;
        }
        .alarm-title {
            font-weight: bold;
            margin-bottom: 10px;
        }
        .detection-list {
            margin-top: 15px;
        }
        .detection-item {
            background: rgba(255,255,255,0.05);
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 8px;
        }
        .detection-item .label {
            color: #2ecc71;
            font-weight: bold;
        }
        .detection-item .fire {
            color: #e74c3c;
        }
        .loading {
            text-align: center;
            padding: 40px;
            display: none;
        }
        .loading.show { display: block; }
        .spinner {
            border: 4px solid rgba(255,255,255,0.1);
            border-top-color: #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 消防器材检测 <span class="mode-badge">纯YOLO模式</span></h1>
        <p class="subtitle">上传动火作业现场图片，检测消防器材配备情况</p>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📷</div>
            <p>点击或拖拽图片到此处上传</p>
            <p style="color: #666; font-size: 0.9em; margin-top: 10px;">支持 PNG, JPG, JPEG, BMP, WEBP</p>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>正在检测中...</p>
        </div>
        
        <div class="results" id="results" style="display: none;">
            <div class="result-card">
                <h3>📸 原始图片</h3>
                <img id="originalImage" src="" alt="原始图片">
            </div>
            <div class="result-card">
                <h3>🎯 检测结果</h3>
                <img id="resultImage" src="" alt="检测结果">
                <div id="alarmBox" class="alarm-box">
                    <div class="alarm-title" id="alarmTitle">检测结果</div>
                    <div id="alarmMessage"></div>
                </div>
                <div class="detection-list" id="detectionList"></div>
            </div>
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        async function handleFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            loading.classList.add('show');
            results.style.display = 'none';
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert('检测失败: ' + data.error);
                    return;
                }
                
                displayResults(data);
                
            } catch (error) {
                alert('请求失败: ' + error.message);
            } finally {
                loading.classList.remove('show');
            }
        }
        
        function displayResults(data) {
            results.style.display = 'grid';
            
            document.getElementById('originalImage').src = '/static/' + data.original;
            document.getElementById('resultImage').src = '/static/' + data.result_image;
            
            const alarmBox = document.getElementById('alarmBox');
            const alarmTitle = document.getElementById('alarmTitle');
            const alarmMessage = document.getElementById('alarmMessage');
            
            if (data.is_alarm) {
                alarmBox.classList.remove('safe');
                alarmTitle.textContent = '⚠️ ' + (data.alarm_details?.boxes?.[0]?.type || '检测到违规');
                alarmMessage.textContent = data.alarm_details?.boxes?.[0]?.msg || '请检查消防器材配备情况';
            } else {
                alarmBox.classList.add('safe');
                alarmTitle.textContent = '✅ 检测通过';
                alarmMessage.textContent = '消防器材配备符合要求';
            }
            
            // 显示检测详情
            const detectionList = document.getElementById('detectionList');
            detectionList.innerHTML = '';
            
            const detection = data.detection;
            
            if (detection.equipment && detection.equipment.length > 0) {
                detection.equipment.forEach(item => {
                    const div = document.createElement('div');
                    div.className = 'detection-item';
                    div.innerHTML = `<span class="label">✅ ${item.label_cn}</span> - 置信度: ${(item.conf * 100).toFixed(1)}%`;
                    detectionList.appendChild(div);
                });
            }
            
            if (detection.fire_zones && detection.fire_zones.length > 0) {
                detection.fire_zones.forEach(item => {
                    const div = document.createElement('div');
                    div.className = 'detection-item';
                    div.innerHTML = `<span class="label fire">🔥 ${item.label_cn || '动火点'}</span>`;
                    detectionList.appendChild(div);
                });
            }
            
            if (detection.equipment_count === 0 && detection.fire_zone_count === 0) {
                const div = document.createElement('div');
                div.className = 'detection-item';
                div.innerHTML = '未检测到消防器材或动火点';
                detectionList.appendChild(div);
            }
        }
    </script>
</body>
</html>
'''


def create_template():
    """创建前端模板文件"""
    template_dir = 'templates'
    os.makedirs(template_dir, exist_ok=True)
    
    template_path = os.path.join(template_dir, 'fire_index.html')
    if not os.path.exists(template_path):
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(FIRE_INDEX_HTML)
        print(f"✅ 已创建模板文件: {template_path}")


if __name__ == '__main__':
    create_template()
    load_service()
    
    print("\n" + "=" * 50)
    print("🔥 消防器材检测服务 - 纯YOLO模式")
    print("=" * 50)
    print(f"📍 访问地址: http://localhost:5001")
    print(f"📍 API 端点: POST http://localhost:5001/detect")
    print("=" * 50 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
