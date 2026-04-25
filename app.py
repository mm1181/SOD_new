import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from models import SODModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 创建必要的目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型路径
MODEL_PATH = 'outputs/best_model.pth'
PVT_WEIGHTS_PATH = 'weights/pvt_v2_b2.safetensors'

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 全局模型实例
model = None

def load_model():
    global model
    pvt_path = PVT_WEIGHTS_PATH if os.path.exists(PVT_WEIGHTS_PATH) else None
    model = SODModel(pretrained_resnet=False, pretrained_pvt_path=pvt_path)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'ema' in checkpoint and checkpoint['ema'] is not None:
            # 如果有 EMA 权重，优先使用
            model.load_state_dict(checkpoint['ema'])
        else:
            model.load_state_dict(checkpoint)
        print(f"模型加载成功: {MODEL_PATH}")
    else:
        print(f"警告: 模型文件不存在 {MODEL_PATH}")
    
    model.to(device)
    model.eval()

def predict(image_path):
    """对输入图像进行显著目标检测"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    # 预处理
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        saliency_list, edge_1_4, edge_1_8 = model(input_tensor)
        saliency_map = torch.sigmoid(saliency_list[0])
        edge_map = torch.sigmoid((edge_1_4 + edge_1_8) / 2)
    
    # 上采样到原始尺寸
    saliency_map = F.interpolate(saliency_map, size=(original_size[1], original_size[0]), 
                                  mode='bilinear', align_corners=False)
    edge_map = F.interpolate(edge_map, size=(original_size[1], original_size[0]), 
                              mode='bilinear', align_corners=False)
    
    # 转换为numpy数组
    saliency_np = saliency_map.squeeze().cpu().numpy()
    edge_np = edge_map.squeeze().cpu().numpy()
    
    # 生成唯一文件名
    file_id = str(uuid.uuid4())[:8]
    
    # 保存显著图
    saliency_img = Image.fromarray((saliency_np * 255).astype(np.uint8))
    saliency_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_saliency.png')
    saliency_img.save(saliency_path)
    
    # 保存边缘图
    edge_img = Image.fromarray((edge_np * 255).astype(np.uint8))
    edge_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_edge.png')
    edge_img.save(edge_path)
    
    # 创建叠加图像
    image_np = np.array(image)
    overlay = image_np.copy().astype(np.float32)
    mask = saliency_np > 0.5
    overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    overlay_img = Image.fromarray(overlay.astype(np.uint8))
    overlay_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}_overlay.png')
    overlay_img.save(overlay_path)
    
    return {
        'saliency': f'results/{file_id}_saliency.png',
        'edge': f'results/{file_id}_edge.png',
        'overlay': f'results/{file_id}_overlay.png'
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        # 保存上传的文件
        file_id = str(uuid.uuid4())[:8]
        ext = os.path.splitext(file.filename)[1]
        filename = f'{file_id}{ext}'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 进行预测
        results = predict(filepath)
        results['original'] = f'uploads/{filename}'
        
        return jsonify(results)
    
    return jsonify({'error': '不支持的文件格式'}), 400

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
