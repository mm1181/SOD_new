"""
消防器材检测服务类

参考 ai_service.py 的结构设计，实现：
1. YOLO 模型懒加载
2. 消防器材检测（灭火器、消防水桶、灭火毯）
3. 动火点检测（纯 YOLO 模式）或接收 SOD 分割结果（混合模式）
4. 报警冷却控制
"""

import cv2
import os
import time
import numpy as np


class FireEquipmentService:
    """消防器材检测服务"""
    
    def __init__(self, model_path="fire_detection/weights/fire_equipment.pt",
                 cooldown_seconds=5, shared_cooldown_map=None):
        """
        初始化服务
        
        Args:
            model_path: YOLO 模型权重路径
            cooldown_seconds: 报警冷却时间（秒）
            shared_cooldown_map: 共享的冷却时间映射（用于多实例共享）
        """
        self.model_path = model_path
        self.model = None
        self.cooldown_seconds = cooldown_seconds
        self.last_alarm_time_map = shared_cooldown_map if shared_cooldown_map is not None else {}
        
        # 检测类别映射（对应 data.yaml 中的顺序）
        # names: fire_bucket, fire_blanket, fire_extinguisher, fire, smoke, spark
        self.class_names = {
            0: "fire_bucket",          # 消防水桶
            1: "fire_blanket",         # 灭火毯
            2: "fire_extinguisher",    # 灭火器
            3: "fire",                 # 火焰（动火点）
            4: "smoke",                # 烟雾（动火点）
            5: "spark",                # 火花（动火点）
        }
        
        # 类别中文名称（用于报警信息）
        self.class_names_cn = {
            "fire_extinguisher": "灭火器",
            "fire_bucket": "消防水桶",
            "fire_blanket": "灭火毯",
            "fire": "火焰",
            "smoke": "烟雾",
            "spark": "火花",
            "fire_zone_sod": "动火区域(SOD)",
        }
        
        # 消防器材类别列表
        self.equipment_classes = ["fire_extinguisher", "fire_bucket", "fire_blanket"]
        
        # 动火点类别列表（fire, smoke, spark 都视为动火点）
        self.fire_zone_classes = ["fire", "smoke", "spark"]
        
        # 缺失计数器（用于连续帧判断，避免误报）
        self.missing_counter = 0
        self.MISSING_THRESHOLD = int(os.getenv("FIRE_MISSING_THRESHOLD", "3"))
        
        # 调试模式
        self.debug_mode = os.getenv("FIRE_DEBUG", "0") == "1"
    
    def _load_model_safe(self):
        """
        懒加载 YOLO 模型
        
        Returns:
            bool: 加载是否成功
        """
        if self.model is not None:
            return True
        
        try:
            print("⏳ [消防检测] 正在加载 YOLO 模型...")
            base_dir = os.getcwd()
            full_path = os.path.join(base_dir, self.model_path)
            
            if not os.path.exists(full_path):
                print(f"❌ [错误] 找不到模型文件: {full_path}")
                return False
            
            from ultralytics import YOLO
            loaded_model = YOLO(full_path)
            
            # 根据环境选择设备
            device = "cuda" if self._cuda_available() else "cpu"
            loaded_model.to(device)
            
            self.model = loaded_model
            print(f"✅ [消防检测] 模型加载完成 (设备: {device})")
            return True
            
        except Exception as e:
            print(f"❌ [严重错误] 模型加载失败: {e}")
            return False
    
    def _cuda_available(self):
        """检查 CUDA 是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False
    
    def _fire_equipment_detect(self, frame, conf=0.5, use_sod=False, sod_result=None):
        """
        消防器材检测核心方法
        
        Args:
            frame: 输入图像 (BGR 格式，numpy array)
            conf: 置信度阈值
            use_sod: 是否使用 SOD 结果替代 YOLO 的动火点检测
            sod_result: SOD 模型的分割结果（动火区域 mask，numpy array）
        
        Returns:
            dict: 检测结果，包含 equipment 和 fire_zones
            None: 检测失败
        """
        if not self._load_model_safe():
            return None
        
        # YOLO 推理
        results = self.model(frame, conf=conf, verbose=False)[0]
        
        equipment_boxes = []  # 消防器材
        fire_zones = []       # 动火区域
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names.get(cls_id, self.class_names.get(cls_id, "unknown"))
            conf_val = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            
            item = {
                "label": cls_name,
                "label_cn": self.class_names_cn.get(cls_name, cls_name),
                "conf": conf_val,
                "coords": coords,
            }
            
            if cls_name in self.equipment_classes:
                equipment_boxes.append(item)
            elif cls_name in self.fire_zone_classes and not use_sod:
                # 纯 YOLO 模式：fire, smoke, spark 都视为动火点
                fire_zones.append(item)
        
        # SOD + YOLO 模式：动火区域由 SOD 提供
        if use_sod and sod_result is not None:
            fire_zones = self._extract_fire_zone_from_sod(sod_result)
        
        # 调试输出
        if self.debug_mode or equipment_boxes or fire_zones:
            print(f"🔍 [消防检测] 器材: {len(equipment_boxes)} | 动火区域: {len(fire_zones)}")
            for e in equipment_boxes:
                print(f"   ✅ {e['label_cn']} conf={e['conf']:.3f}")
            for z in fire_zones:
                print(f"   🔥 {z.get('label_cn', z['label'])} coords={[int(x) for x in z['coords']]}")
        
        return {
            "equipment": equipment_boxes,
            "fire_zones": fire_zones,
        }
    
    def _extract_fire_zone_from_sod(self, sod_mask):
        """
        从 SOD 分割结果提取动火区域
        
        Args:
            sod_mask: SOD 输出的显著性图 (numpy array, 值域 0-1)
        
        Returns:
            list: 动火区域列表
        """
        # 二值化
        binary = (sod_mask > 0.5).astype(np.uint8) * 255
        
        # 形态学操作：去噪 + 填充
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        zones = []
        min_area = 100  # 最小面积阈值，过滤噪声
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 计算轮廓中心
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            zones.append({
                "label": "fire_zone_sod",
                "label_cn": "动火区域(SOD)",
                "coords": [x, y, x + w, y + h],
                "center": [cx, cy],
                "area": area,
                "contour": cnt,  # 保留轮廓用于精确判断
            })
        
        return zones
    
    def _check_cooldown_and_alarm(self, alarm_type, msg, score, coords, skip_cooldown=False):
        """
        报警冷却控制
        
        Args:
            alarm_type: 报警类型
            msg: 报警消息
            score: 置信度分数
            coords: 坐标 [x1, y1, x2, y2]
            skip_cooldown: 是否跳过冷却检查（图片检测模式应设为True）
        
        Returns:
            tuple: (是否报警, 报警详情)
        """
        now = time.time()
        
        # 规范化冷却 key
        cooldown_key = alarm_type
        if "消防器材" in alarm_type:
            cooldown_key = "FIRE_EQUIPMENT_COOLDOWN"
        elif "灭火毯" in alarm_type:
            cooldown_key = "FIRE_BLANKET_COOLDOWN"
        elif "灭火器" in alarm_type:
            cooldown_key = "EXTINGUISHER_COOLDOWN"
        
        last = self.last_alarm_time_map.get(cooldown_key, 0.0)
        
        # 特定类型使用更长的冷却时间
        current_cooldown = self.cooldown_seconds
        is_long_cooldown = cooldown_key in ["FIRE_EQUIPMENT_COOLDOWN"]
        if is_long_cooldown:
            current_cooldown = 300  # 5分钟
        
        # 跳过冷却检查（图片检测模式）或超过冷却时间
        if skip_cooldown or (now - last > current_cooldown):
            if not skip_cooldown:
                # 只有非跳过模式才更新冷却时间
                if is_long_cooldown:
                    print(f"✅ [冷却锁定] 类型:{alarm_type} 已进入5分钟锁定期")
                self.last_alarm_time_map[cooldown_key] = now
            
            print(f"🚨 [消防检测] 报警已发出! ({alarm_type})")
            
            data = {
                "alarm": True,
                "boxes": [
                    {
                        "type": alarm_type,
                        "msg": msg,
                        "score": score,
                        "coords": coords
                    }
                ]
            }
            
            return True, data
        
        return False, None
    
    def _check_cooldown_and_multi_alarm(self, alarm_type, boxes):
        """
        多目标报警冷却控制
        
        Args:
            alarm_type: 报警类型
            boxes: 多个报警框列表
        
        Returns:
            tuple: (是否报警, 报警详情)
        """
        now = time.time()
        cooldown_key = alarm_type
        last = self.last_alarm_time_map.get(cooldown_key, 0.0)
        
        if now - last > self.cooldown_seconds:
            self.last_alarm_time_map[cooldown_key] = now
            data = {"alarm": True, "boxes": boxes}
            print(f"🚨 [消防检测] 报警已发出! ({alarm_type}) [{len(boxes)}个目标]")
            return True, data
        
        return False, None
    
    # ===== 对外接口 =====
    
    def detect_fire_equipment(self, frame, conf=0.5, skip_cooldown=True):
        """
        纯 YOLO 模式检测
        
        Args:
            frame: 输入图像
            conf: 置信度阈值
            skip_cooldown: 是否跳过冷却检查（图片检测默认True，视频流设为False）
        
        Returns:
            tuple: (是否报警, 报警详情)
        """
        return self._detect_and_judge(frame, conf=conf, use_sod=False, sod_result=None, skip_cooldown=skip_cooldown)
    
    def detect_fire_equipment_with_sod(self, frame, sod_result, conf=0.5, skip_cooldown=True):
        """
        SOD + YOLO 模式检测
        
        Args:
            frame: 输入图像
            sod_result: SOD 模型的分割结果
            conf: 置信度阈值
            skip_cooldown: 是否跳过冷却检查（图片检测默认True，视频流设为False）
        
        Returns:
            tuple: (是否报警, 报警详情)
        """
        return self._detect_and_judge(frame, conf=conf, use_sod=True, sod_result=sod_result, skip_cooldown=skip_cooldown)
    
    def _detect_and_judge(self, frame, conf, use_sod, sod_result, skip_cooldown=True):
        """
        检测 + 违规判定
        
        Args:
            frame: 输入图像
            conf: 置信度阈值
            use_sod: 是否使用 SOD 模式
            sod_result: SOD 分割结果
            skip_cooldown: 是否跳过冷却检查
        
        Returns:
            tuple: (是否报警, 报警详情)
        """
        from fire_detection import fire_features
        
        detect_result = self._fire_equipment_detect(
            frame, conf=conf, use_sod=use_sod, sod_result=sod_result
        )
        
        if detect_result is None:
            return False, None
        
        # 调用违规判定逻辑，传递 skip_cooldown 参数
        return fire_features.judge_fire_equipment_violation(self, detect_result, skip_cooldown=skip_cooldown)
    
    def get_raw_detection(self, frame, conf=0.5, use_sod=False, sod_result=None):
        """
        获取原始检测结果（不进行违规判定）
        
        用于调试或自定义后处理
        
        Args:
            frame: 输入图像
            conf: 置信度阈值
            use_sod: 是否使用 SOD 模式
            sod_result: SOD 分割结果
        
        Returns:
            dict: 原始检测结果
        """
        return self._fire_equipment_detect(
            frame, conf=conf, use_sod=use_sod, sod_result=sod_result
        )
    
    def draw_detection_result(self, frame, detect_result):
        """
        在图像上绘制检测结果
        
        Args:
            frame: 输入图像
            detect_result: 检测结果字典
        
        Returns:
            numpy.ndarray: 绘制后的图像
        """
        if detect_result is None:
            return frame
        
        draw_frame = frame.copy()
        
        # 绘制消防器材（绿色）
        for item in detect_result.get("equipment", []):
            coords = item["coords"]
            x1, y1, x2, y2 = map(int, coords)
            label = f"{item['label_cn']} {item['conf']:.2f}"
            
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(draw_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制动火区域（红色）
        for item in detect_result.get("fire_zones", []):
            coords = item["coords"]
            x1, y1, x2, y2 = map(int, coords)
            label = item.get("label_cn", item["label"])
            
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(draw_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 如果有轮廓，绘制轮廓
            if "contour" in item:
                cv2.drawContours(draw_frame, [item["contour"]], -1, (0, 0, 255), 1)
        
        return draw_frame
