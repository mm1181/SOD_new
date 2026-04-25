"""
消防器材检测模块

支持两种运行模式：
1. 纯 YOLO 模式：独立完成消防器材检测和违规判定
2. SOD + YOLO 模式：YOLO 检测消防器材，SOD 提供动火区域分割
"""

from .fire_service import FireEquipmentService

__all__ = ['FireEquipmentService']
