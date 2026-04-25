"""
消防器材违规判定逻辑

依据标准：《建设工程施工现场消防安全技术规范》（GB 50720-2011）

报警逻辑（统一纯YOLO模式和混合模式）：
- 无动火点 → 不报警
- 有动火点 + 至少满足一个消防措施 → 不报警
- 有动火点 + 不满足任何消防措施 → 报警

消防措施定义：
1. 半径10m内存在至少1个灭火器
2. 半径10m内存在至少1个消防水桶
3. 存在灭火毯且覆盖动火点下方区域
"""

import math


# 距离阈值（像素），用于估算10m范围
# 1000像素 = 10米
DISTANCE_THRESHOLD_PX = 1000


def judge_fire_equipment_violation(service, detect_result, skip_cooldown=True):
    """
    违规判定主函数（统一逻辑）

    Args:
        service: FireEquipmentService 实例（用于调用冷却控制）
        detect_result: 检测结果字典，包含 equipment 和 fire_zones
        skip_cooldown: 是否跳过冷却检查（图片检测默认True，视频流设为False）

    Returns:
        tuple: (是否报警, 报警详情)
    """
    equipment = detect_result.get("equipment", [])
    fire_zones = detect_result.get("fire_zones", [])
    
    # 调试输出
    print(f"📋 [违规判定] 器材数量: {len(equipment)}, 动火点数量: {len(fire_zones)}")
    
    # ===== 情况1：无动火点 → 不报警 =====
    if not fire_zones:
        print("✅ [违规判定] 无动火点，不报警")
        return False, None
    
    # ===== 情况2/3：有动火点，检查是否满足消防措施 =====
    
    # 分类器材
    extinguishers = [e for e in equipment if e["label"] == "fire_extinguisher"]
    buckets = [e for e in equipment if e["label"] == "fire_bucket"]
    blankets = [e for e in equipment if e["label"] == "fire_blanket"]
    
    print(f"📋 [违规判定] 灭火器: {len(extinguishers)}, 消防水桶: {len(buckets)}, 灭火毯: {len(blankets)}")
    
    # 对每个动火点检查是否满足至少一个消防措施
    for idx, zone in enumerate(fire_zones):
        measure_satisfied = False
        zone_label = zone.get("label_cn", zone.get("label", "动火点"))
        print(f"🔥 [违规判定] 检查动火点 {idx+1}: {zone_label}")
        
        # 措施1：半径10m内存在至少1个灭火器
        if _has_nearby_equipment(extinguishers, zone, threshold_px=DISTANCE_THRESHOLD_PX):
            print(f"   ✅ 措施1满足: 附近有灭火器")
            measure_satisfied = True
        
        # 措施2：半径10m内存在至少1个消防水桶
        if _has_nearby_equipment(buckets, zone, threshold_px=DISTANCE_THRESHOLD_PX):
            print(f"   ✅ 措施2满足: 附近有消防水桶")
            measure_satisfied = True
        
        # 措施3：存在灭火毯且覆盖动火点下方区域
        if _is_blanket_covering_correctly(blankets, zone):
            print(f"   ✅ 措施3满足: 灭火毯正确覆盖")
            measure_satisfied = True
        
        # 如果该动火点不满足任何消防措施 → 报警
        if not measure_satisfied:
            print(f"   ❌ 动火点 {idx+1} 不满足任何消防措施，触发报警")
            return service._check_cooldown_and_alarm(
                alarm_type="消防措施不足",
                msg="动火作业现场未满足消防安全要求：半径10m内无灭火器/消防水桶，且无正确使用的灭火毯，违反GB 50720-2011规定",
                score=1.0,
                coords=zone["coords"],
                skip_cooldown=skip_cooldown
            )
        else:
            print(f"   ✅ 动火点 {idx+1} 满足消防措施")
    
    # 所有动火点都满足至少一个消防措施 → 不报警
    print("✅ [违规判定] 所有动火点都满足消防措施，不报警")
    return False, None


def _has_nearby_equipment(equipment_list, fire_zone, threshold_px=DISTANCE_THRESHOLD_PX):
    """
    判断是否有器材在动火点附近（半径10m内）
    
    Args:
        equipment_list: 器材检测结果列表
        fire_zone: 动火区域
        threshold_px: 距离阈值（像素），默认300px估算为10m
    
    Returns:
        bool: 是否有器材在附近
    """
    if not equipment_list:
        return False
    
    zx1, zy1, zx2, zy2 = fire_zone["coords"]
    zone_cx, zone_cy = (zx1 + zx2) / 2, (zy1 + zy2) / 2
    
    for e in equipment_list:
        ex1, ey1, ex2, ey2 = e["coords"]
        ecx, ecy = (ex1 + ex2) / 2, (ey1 + ey2) / 2
        
        # 计算中心点距离
        dist = math.sqrt((ecx - zone_cx) ** 2 + (ecy - zone_cy) ** 2)
        
        if dist < threshold_px:
            return True
    
    return False


def _is_blanket_covering_correctly(blankets, fire_zone):
    """
    判断是否存在灭火毯且正确覆盖动火点下方区域
    
    判定逻辑：
    1. 灭火毯的水平范围应覆盖动火点中心
    2. 灭火毯应位于动火点下方（y坐标更大）
    3. 灭火毯与动火点距离不能太远
    
    Args:
        blankets: 灭火毯检测结果列表
        fire_zone: 动火区域
    
    Returns:
        bool: 是否正确覆盖
    """
    if not blankets:
        return False
    
    zx1, zy1, zx2, zy2 = fire_zone["coords"]
    zone_center_x = (zx1 + zx2) / 2
    zone_center_y = (zy1 + zy2) / 2
    zone_bottom_y = zy2  # 动火点底部
    zone_height = zy2 - zy1
    
    for b in blankets:
        bx1, by1, bx2, by2 = b["coords"]
        blanket_center_y = (by1 + by2) / 2
        
        # 条件1：灭火毯水平范围覆盖动火点中心
        horizontal_cover = bx1 < zone_center_x < bx2
        
        # 条件2：灭火毯位于动火点下方（中心点比较）
        vertical_below = blanket_center_y > zone_center_y
        
        # 条件3：距离不能太远（灭火毯顶部与动火点底部的距离不超过动火点高度的2倍）
        vertical_distance = by1 - zone_bottom_y
        close_enough = vertical_distance < zone_height * 2
        
        if horizontal_cover and vertical_below and close_enough:
            return True
    
    return False


# ===== 辅助函数 =====

def _calculate_distance(box1, box2):
    """
    计算两个框中心点之间的距离
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        float: 距离（像素）
    """
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def _calculate_iou(box1, box2):
    """
    计算两个框的 IoU（交并比）
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        float: IoU 值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def set_distance_threshold(threshold_px):
    """
    设置距离阈值（用于调整10m范围的像素估算）
    
    Args:
        threshold_px: 新的距离阈值（像素）
    """
    global DISTANCE_THRESHOLD_PX
    DISTANCE_THRESHOLD_PX = threshold_px
