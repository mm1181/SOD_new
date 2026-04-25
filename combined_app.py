"""
消防器材检测 - SOD + YOLO 联合运行入口

功能：
1. SOD 模型检测动火区域（显著性分割）
2. YOLO 模型检测消防器材
3. 两者结合进行违规判定

运行方式：
    python combined_app.py

API 端点：
    POST /detect - 上传图片进行联合检测
    POST /detect_sod_only - 仅 SOD 检测
    POST /detect_yolo_only - 仅 YOLO 检测
    GET / - 检测页面
"""

import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from models import SODModel
from fire_detection import FireEquipmentService

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/combined_uploads'
app.config['RESULT_FOLDER'] = 'static/combined_results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 创建必要的目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型路径
SOD_MODEL_PATH = 'outputs/best_model.pth'
PVT_WEIGHTS_PATH = 'weights/pvt_v2_b2.safetensors'

# 图像预处理（SOD 模型）
sod_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 全局模型实例
sod_model = None
fire_service = None


def load_models():
    """加载 SOD 和 YOLO 模型"""
    global sod_model, fire_service
    
    # 加载 SOD 模型
    print("⏳ 正在加载 SOD 模型...")
    pvt_path = PVT_WEIGHTS_PATH if os.path.exists(PVT_WEIGHTS_PATH) else None
    sod_model = SODModel(pretrained_resnet=False, pretrained_pvt_path=pvt_path)
    
    if os.path.exists(SOD_MODEL_PATH):
        checkpoint = torch.load(SOD_MODEL_PATH, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            sod_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            sod_model.load_state_dict(checkpoint['state_dict'])
        elif 'ema' in checkpoint and checkpoint['ema'] is not None:
            sod_model.load_state_dict(checkpoint['ema'])
        else:
            sod_model.load_state_dict(checkpoint)
        print(f"✅ SOD 模型加载成功: {SOD_MODEL_PATH}")
    else:
        print(f"⚠️ 警告: SOD 模型文件不存在 {SOD_MODEL_PATH}")
    
    sod_model.to(device)
    sod_model.eval()
    
    # 加载消防检测服务
    print("⏳ 正在初始化消防检测服务...")
    fire_service = FireEquipmentService(
        model_path="fire_detection/weights/fire_equipment.pt",
        cooldown_seconds=5
    )
    print("✅ 消防检测服务初始化完成")


def get_sod_mask(image):
    """
    使用 SOD 模型获取显著性分割结果
    
    Args:
        image: PIL Image 或 numpy array (BGR)
    
    Returns:
        numpy.ndarray: 显著性 mask (0-1 范围)
    """
    # 转换为 PIL Image
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        image_pil = image
    
    original_size = image_pil.size  # (W, H)
    
    # 预处理
    input_tensor = sod_transform(image_pil).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        saliency_list, edge_1_4, edge_1_8 = sod_model(input_tensor)
        saliency_map = torch.sigmoid(saliency_list[0])
    
    # 上采样到原始尺寸
    saliency_map = F.interpolate(
        saliency_map, 
        size=(original_size[1], original_size[0]),  # (H, W)
        mode='bilinear', 
        align_corners=False
    )
    
    # 转换为 numpy
    sod_mask = saliency_map.squeeze().cpu().numpy()
    
    return sod_mask


def process_image_combined(image_path):
    """
    SOD + YOLO 联合检测
    
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
    
    # Step 1: SOD 获取动火区域
    sod_mask = get_sod_mask(image)
    
    # Step 2: YOLO + SOD 联合检测
    is_alarm, alarm_details = fire_service.detect_fire_equipment_with_sod(
        image, sod_mask, conf=0.5
    )
    
    # 获取原始检测结果
    raw_result = fire_service.get_raw_detection(
        image, conf=0.5, use_sod=True, sod_result=sod_mask
    )
    
    # 生成唯一文件名
    file_id = str(uuid.uuid4())[:8]
    
    # 保存 SOD 显著性图
    sod_img = Image.fromarray((sod_mask * 255).astype(np.uint8))
    sod_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_sod.png')
    sod_img.save(sod_path)
    
    # 绘制检测结果
    result_image = fire_service.draw_detection_result(image, raw_result)
    
    # 叠加 SOD 热力图
    heatmap = cv2.applyColorMap((sod_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(result_image, 0.7, heatmap, 0.3, 0)
    
    # 保存结果图片
    result_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_result.jpg')
    cv2.imwrite(result_path, result_image)
    
    overlay_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_overlay.jpg')
    cv2.imwrite(overlay_path, overlay)
    
    # 构建返回结果
    result = {
        "file_id": file_id,
        "mode": "sod_yolo_combined",
        "sod_image": f'combined_results/{file_id}_sod.png',
        "result_image": f'combined_results/{file_id}_result.jpg',
        "overlay_image": f'combined_results/{file_id}_overlay.jpg',
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


def process_image_sod_only(image_path):
    """仅 SOD 检测"""
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "无法读取图片"}
    
    sod_mask = get_sod_mask(image)
    
    file_id = str(uuid.uuid4())[:8]
    
    # 保存显著性图
    sod_img = Image.fromarray((sod_mask * 255).astype(np.uint8))
    sod_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_sod.png')
    sod_img.save(sod_path)
    
    # 创建叠加图
    heatmap = cv2.applyColorMap((sod_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    overlay_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_overlay.jpg')
    cv2.imwrite(overlay_path, overlay)
    
    return {
        "file_id": file_id,
        "mode": "sod_only",
        "sod_image": f'combined_results/{file_id}_sod.png',
        "overlay_image": f'combined_results/{file_id}_overlay.jpg',
    }


def process_image_yolo_only(image_path):
    """仅 YOLO 检测"""
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "无法读取图片"}
    
    is_alarm, alarm_details = fire_service.detect_fire_equipment(image, conf=0.5)
    raw_result = fire_service.get_raw_detection(image, conf=0.5)
    
    file_id = str(uuid.uuid4())[:8]
    
    result_image = fire_service.draw_detection_result(image, raw_result)
    result_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_result.jpg')
    cv2.imwrite(result_path, result_image)
    
    return {
        "file_id": file_id,
        "mode": "yolo_only",
        "result_image": f'combined_results/{file_id}_result.jpg',
        "is_alarm": is_alarm,
        "alarm_details": alarm_details,
        "detection": {
            "equipment_count": len(raw_result.get("equipment", [])) if raw_result else 0,
            "fire_zone_count": len(raw_result.get("fire_zones", [])) if raw_result else 0,
        }
    }


@app.route('/')
def index():
    """检测页面"""
    return render_template('combined_index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """SOD + YOLO 联合检测 API"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({'error': f'不支持的文件格式: {ext}'}), 400
    
    file_id = str(uuid.uuid4())[:8]
    filename = f'{file_id}{ext}'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    result = process_image_combined(filepath)
    result['original'] = f'combined_uploads/{filename}'
    
    return jsonify(result)


@app.route('/detect_sod_only', methods=['POST'])
def detect_sod_only():
    """仅 SOD 检测 API"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    ext = os.path.splitext(file.filename)[1].lower()
    
    file_id = str(uuid.uuid4())[:8]
    filename = f'{file_id}{ext}'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    result = process_image_sod_only(filepath)
    result['original'] = f'combined_uploads/{filename}'
    
    return jsonify(result)


@app.route('/detect_yolo_only', methods=['POST'])
def detect_yolo_only():
    """仅 YOLO 检测 API"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    ext = os.path.splitext(file.filename)[1].lower()
    
    file_id = str(uuid.uuid4())[:8]
    filename = f'{file_id}{ext}'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    result = process_image_yolo_only(filepath)
    result['original'] = f'combined_uploads/{filename}'
    
    return jsonify(result)


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'service': 'fire_detection',
        'mode': 'sod_yolo_combined',
        'device': str(device)
    })


@app.route('/static/<path:filename>')
def static_files(filename):
    """静态文件服务"""
    return send_from_directory('static', filename)


# 前端模板
COMBINED_INDEX_HTML = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>消防器材检测 - SOD+YOLO联合模式</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 10px; font-size: 2em; }
        .subtitle { text-align: center; color: #888; margin-bottom: 20px; }
        .mode-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        .mode-selector {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 25px;
        }
        .mode-btn {
            padding: 10px 25px;
            border: 2px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.05);
            color: #fff;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .mode-btn:hover { border-color: #667eea; background: rgba(102, 126, 234, 0.2); }
        .mode-btn.active {
            border-color: #667eea;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .upload-area {
            background: rgba(255,255,255,0.05);
            border: 2px dashed rgba(255,255,255,0.2);
            border-radius: 16px;
            padding: 50px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 25px;
        }
        .upload-area:hover { border-color: #667eea; background: rgba(102, 126, 234, 0.1); }
        .upload-icon { font-size: 48px; margin-bottom: 15px; }
        #fileInput { display: none; }
        .results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .result-card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 15px;
        }
        .result-card h3 { margin-bottom: 12px; color: #667eea; font-size: 1em; }
        .result-card img { width: 100%; border-radius: 8px; }
        .alarm-box {
            background: rgba(231, 76, 60, 0.2);
            border: 1px solid #e74c3c;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }
        .alarm-box.safe { background: rgba(46, 204, 113, 0.2); border-color: #2ecc71; }
        .loading { text-align: center; padding: 40px; display: none; }
        .loading.show { display: block; }
        .spinner {
            border: 4px solid rgba(255,255,255,0.1);
            border-top-color: #667eea;
            border-radius: 50%;
            width: 40px; height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .info-box {
            background: rgba(102, 126, 234, 0.1);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .info-box h4 { color: #667eea; margin-bottom: 8px; }
        .info-box p { color: #aaa; font-size: 0.9em; line-height: 1.6; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 消防器材检测 <span class="mode-badge">SOD+YOLO联合</span></h1>
        <p class="subtitle">结合显著性目标检测与YOLO目标检测，精准识别动火区域和消防器材</p>
        
        <div class="info-box">
            <h4>💡 模式说明</h4>
            <p>
                <strong>联合模式</strong>：SOD 检测动火区域（火焰/焊接点），YOLO 检测消防器材，两者结合判断违规<br>
                <strong>仅SOD</strong>：显著性目标检测，用于定位动火区域<br>
                <strong>仅YOLO</strong>：目标检测，同时检测消防器材和动火点
            </p>
        </div>
        
        <div class="mode-selector">
            <button class="mode-btn active" data-mode="combined">🔗 联合模式</button>
            <button class="mode-btn" data-mode="sod">🎯 仅SOD</button>
            <button class="mode-btn" data-mode="yolo">📦 仅YOLO</button>
        </div>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📷</div>
            <p>点击或拖拽图片到此处上传</p>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>正在检测中...</p>
        </div>
        
        <div class="results" id="results" style="display: none;"></div>
    </div>
    
    <script>
        let currentMode = 'combined';
        const modeEndpoints = {
            'combined': '/detect',
            'sod': '/detect_sod_only',
            'yolo': '/detect_yolo_only'
        };
        
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentMode = btn.dataset.mode;
            });
        });
        
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', e => { e.preventDefault(); });
        uploadArea.addEventListener('drop', e => {
            e.preventDefault();
            if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', e => {
            if (e.target.files.length > 0) handleFile(e.target.files[0]);
        });
        
        async function handleFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            document.getElementById('loading').classList.add('show');
            document.getElementById('results').style.display = 'none';
            
            try {
                const response = await fetch(modeEndpoints[currentMode], {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.error) { alert('检测失败: ' + data.error); return; }
                displayResults(data);
            } catch (error) {
                alert('请求失败: ' + error.message);
            } finally {
                document.getElementById('loading').classList.remove('show');
            }
        }
        
        function displayResults(data) {
            const results = document.getElementById('results');
            results.style.display = 'grid';
            results.innerHTML = '';
            
            // 原图
            if (data.original) {
                results.innerHTML += `
                    <div class="result-card">
                        <h3>📸 原始图片</h3>
                        <img src="/static/${data.original}" alt="原图">
                    </div>
                `;
            }
            
            // SOD 结果
            if (data.sod_image) {
                results.innerHTML += `
                    <div class="result-card">
                        <h3>🎯 SOD 显著性图</h3>
                        <img src="/static/${data.sod_image}" alt="SOD">
                    </div>
                `;
            }
            
            // 检测结果
            if (data.result_image) {
                let alarmHtml = '';
                if (data.is_alarm !== undefined) {
                    const alarmClass = data.is_alarm ? '' : 'safe';
                    const title = data.is_alarm ? 
                        '⚠️ ' + (data.alarm_details?.boxes?.[0]?.type || '检测到违规') :
                        '✅ 检测通过';
                    const msg = data.is_alarm ?
                        (data.alarm_details?.boxes?.[0]?.msg || '请检查消防器材') :
                        '消防器材配备符合要求';
                    alarmHtml = `
                        <div class="alarm-box ${alarmClass}">
                            <strong>${title}</strong><br>${msg}
                        </div>
                    `;
                }
                results.innerHTML += `
                    <div class="result-card">
                        <h3>🎯 检测结果</h3>
                        <img src="/static/${data.result_image}" alt="结果">
                        ${alarmHtml}
                    </div>
                `;
            }
            
            // 叠加图
            if (data.overlay_image) {
                results.innerHTML += `
                    <div class="result-card">
                        <h3>🔥 热力叠加图</h3>
                        <img src="/static/${data.overlay_image}" alt="叠加">
                    </div>
                `;
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
    
    template_path = os.path.join(template_dir, 'combined_index.html')
    if not os.path.exists(template_path):
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(COMBINED_INDEX_HTML)
        print(f"✅ 已创建模板文件: {template_path}")


if __name__ == '__main__':
    create_template()
    load_models()
    
    print("\n" + "=" * 50)
    print("🔥 消防器材检测服务 - SOD+YOLO联合模式")
    print("=" * 50)
    print(f"📍 访问地址: http://localhost:5002")
    print(f"📍 设备: {device}")
    print("=" * 50)
    print("API 端点:")
    print("  POST /detect          - 联合检测")
    print("  POST /detect_sod_only - 仅SOD检测")
    print("  POST /detect_yolo_only- 仅YOLO检测")
    print("=" * 50 + "\n")
    
    app.run(host='0.0.0.0', port=5002, debug=False)
