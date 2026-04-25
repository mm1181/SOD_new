import threading
import time
import cv2
import os
import uuid
import re
import requests
import numpy as np
from datetime import datetime, timedelta
from app.services.ai_service import AIService
from app.models.alarm_records import AlarmRecord
from app.models.video import VideoDevice
from app.core.database import SessionLocal
from app.services import ai_features
from app.services.video_service import VideoService, RECORD_SEGMENT_SECONDS, RECORD_SEGMENT_SAFE_MARGIN_SECONDS
from urllib.parse import urlsplit, urlunsplit, unquote, quote
from PIL import Image, ImageDraw, ImageFont


class AIManager:
    def __init__(self):
        self.active_monitors = {}
        # 全局共享冷却时间映射，解决重启监控或多路干扰导致的冷却失效
        self.global_last_alarm_time = {}
        
        self.ai_service = AIService(shared_cooldown_map=self.global_last_alarm_time)
        self.video_service = VideoService()
        self.base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.static_dir = os.path.join(self.base_dir, "static", "alarms")
        os.makedirs(self.static_dir, exist_ok=True)

        # 算法分发表
        self.algo_handlers = ai_features.get_algo_handlers(self.ai_service)
        print(f"✅ 已加载AI规则: {list(self.algo_handlers.keys())}")

    # =========================
    # 启动监控
    # =========================
    def _normalize_rtsp_path(self, url: str) -> str:
        if not isinstance(url, str):
            return ""
        raw = url.strip()
        if not raw.startswith("rtsp://"):
            return raw

        scheme, _, rest = raw.partition("://")
        if "/" not in rest:
            return raw

        host_part, path_part = rest.split("/", 1)
        return f"{scheme}://{host_part}/" + path_part.lstrip("/")

    def _replace_hik_channel(self, url: str, channel: str) -> str:
        return re.sub(r"/Streaming/Channels/\d+", f"/Streaming/Channels/{channel}", url)

    def _with_double_slash_path(self, url: str) -> str:
        if not isinstance(url, str) or not url.startswith("rtsp://"):
            return url
        scheme, _, rest = url.partition("://")
        if "/" not in rest:
            return url
        host_part, path_part = rest.split("/", 1)
        return f"{scheme}://{host_part}//{path_part.lstrip('/')}"

    def _plan_ai_and_record_rtsp(self, rtsp_url: str):
        """优先将 AI 与录像拆到不同通道，减少部分设备二次 SETUP=500 问题。"""
        normalized = self._normalize_rtsp_path(str(rtsp_url or ""))
        if not normalized:
            return "", ""

        if "/Streaming/Channels/" in normalized:
            ai_url = self._replace_hik_channel(normalized, "101")
            rec_url = self._replace_hik_channel(normalized, "102")
            return ai_url, rec_url

        return normalized, normalized

    def start_monitoring(self, device_id, rtsp_url, algo_type="helmet,smoking"):
        device_id = str(device_id)

        if device_id in self.active_monitors:
            print(f"⚠️ 设备 {device_id} 已经在监控中")
            return False

        ai_rtsp_url, record_rtsp_url = self._plan_ai_and_record_rtsp(rtsp_url)
        monitor_mode = "rtsp"
        ezviz_serial = ""
        ezviz_channel = 1

        if not ai_rtsp_url:
            db = SessionLocal()
            try:
                db_video = None
                if device_id.isdigit():
                    db_video = db.query(VideoDevice).filter(VideoDevice.id == int(device_id)).first()

                if db_video and getattr(db_video, "device_serial", None):
                    ezviz_serial = str(getattr(db_video, "device_serial", "") or "").strip()
                    ezviz_channel = int(getattr(db_video, "channel_no", 1) or 1)
                    monitor_mode = "ezviz_snapshot"
                else:
                    print("❌ AI 启动失败：RTSP 地址为空，且设备未配置萤石云序列号")
                    return False
            finally:
                db.close()

        print(f"--- 启动 AI 监控: {device_id} | 功能: {algo_type} | 模式: {monitor_mode} ---")
        if monitor_mode == "rtsp":
            print(f"🎯 AI拉流地址: {ai_rtsp_url}")
            print(f"💾 录像拉流地址: {record_rtsp_url}")
        else:
            print(f"☁️ 萤石抓图序列号: {ezviz_serial} | 通道: {ezviz_channel}")

        stop_event = threading.Event()
        if monitor_mode == "rtsp":
            thread = threading.Thread(
                target=self._monitor_loop,
                args=(device_id, ai_rtsp_url, record_rtsp_url, algo_type, stop_event),
                daemon=True,
            )
        else:
            thread = threading.Thread(
                target=self._snapshot_monitor_loop,
                args=(device_id, ezviz_serial, ezviz_channel, algo_type, stop_event),
                daemon=True,
            )

        self.active_monitors[device_id] = {
            "stop_event": stop_event,
            "thread": thread,
            "mode": monitor_mode,
        }

        thread.start()
        return True

    def _fetch_ezviz_snapshot_frame(self, device_serial: str, channel_no: int):
        payload = {
            "deviceSerial": device_serial,
            "channelNo": int(channel_no or 1),
        }

        body = None
        for path in ["/api/lapp/device/capture", "/api/lapp/v2/device/capture"]:
            try:
                body = self.video_service._call_ezviz_api(path, payload)
                break
            except Exception:
                body = None

        if body is None:
            return None

        data = body.get("data") or {}
        pic_url = data.get("picUrl") or data.get("url") or data.get("picURL") or ""
        if not pic_url:
            return None

        try:
            response = requests.get(pic_url, timeout=8)
            if response.status_code != 200 or not response.content:
                return None

            np_buf = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def _snapshot_monitor_loop(self, device_id, device_serial, channel_no, algo_type_str, stop_event):
        active_algos = [x.strip() for x in algo_type_str.split(",") if x.strip()]
        # 萤石抓图接口开销较高，默认 1.2s 一帧，必要时可通过环境变量调优。
        interval_seconds = max(0.8, float(os.getenv("AI_EZVIZ_SNAPSHOT_INTERVAL_SECONDS", "1.2")))

        print(f"📸 萤石抓图检测启动: serial={device_serial}, channel={channel_no}, interval={interval_seconds}s")

        while not stop_event.is_set():
            loop_started_at = time.time()
            frame = self._fetch_ezviz_snapshot_frame(device_serial, channel_no)

            if frame is None:
                if stop_event.wait(1.0):
                    break
                continue

            try:
                for algo_key in active_algos:
                    if algo_key not in self.algo_handlers:
                        print(f"⚠️ 未识别算法类型: {algo_key}")
                        continue

                    is_alarm, details = self.algo_handlers[algo_key](frame)

                    if is_alarm:
                        img_path = self._save_alarm_image(frame, device_id, details)
                        self._save_alarm_to_db(device_id, details, img_path)
            except Exception as logic_error:
                print(f"⚠️ 抓图检测逻辑异常: {logic_error}")

            elapsed = time.time() - loop_started_at
            wait_seconds = max(0.0, interval_seconds - elapsed)
            if stop_event.wait(wait_seconds):
                break

        print(f"--- 抓图监控线程已退出: {device_id} ---")

    # =========================
    # 停止监控
    # =========================
    def stop_monitoring(self, device_id):
        device_id = str(device_id)

        if device_id not in self.active_monitors:
            print(f"⚠️ 设备 {device_id} 不在监控中")
            return False

        print(f"--- 停止 AI 监控: {device_id} ---")
        self.active_monitors[device_id]["stop_event"].set()
        del self.active_monitors[device_id]
        return True

    # =========================
    # 主监控循环
    # =========================
    def _build_rtsp_candidates(self, rtsp_url):
        if rtsp_url == 0 or rtsp_url == "0":
            return [0]

        raw = self._normalize_rtsp_path(str(rtsp_url or ""))
        if not raw:
            return []

        candidates = []

        def _push(url):
            if url and url not in candidates:
                candidates.append(url)

        # 候选优先级：101 -> 102 -> 1 -> 当前地址，兼容不同海康通道写法。
        if "/Streaming/Channels/" in raw:
            channel_match = re.search(r"/Streaming/Channels/(\d+)", raw)
            current_channel = channel_match.group(1) if channel_match else ""

            for channel in ["101", "102", "1", current_channel]:
                if not channel:
                    continue
                v = self._replace_hik_channel(raw, channel)
                _push(v)
                _push(self._with_double_slash_path(v))

        _push(raw)
        _push(self._with_double_slash_path(raw))

        # 仅修正路径的重复斜杠，保留原始鉴权串
        if raw.startswith("rtsp://"):
            scheme, _, rest = raw.partition("://")
            if "/" in rest:
                host_part, path_part = rest.split("/", 1)
                fixed_path_url = f"{scheme}://{host_part}/" + path_part.lstrip("/")
                _push(fixed_path_url)

            # 对用户名密码做一次 decode/encode 归一化，兼容 %40 等字符
            try:
                parts = urlsplit(raw)
                host = parts.hostname or ""
                if host:
                    port = f":{parts.port}" if parts.port else ""
                    path = "/" + (parts.path or "").lstrip("/")

                    username = parts.username
                    password = parts.password

                    if username is not None:
                        u_dec = unquote(username)
                        p_dec = unquote(password or "")

                        netloc_encoded = f"{quote(u_dec, safe='')}:{quote(p_dec, safe='')}@{host}{port}"
                        encoded_url = urlunsplit((parts.scheme or "rtsp", netloc_encoded, path, parts.query, parts.fragment))
                        _push(encoded_url)
                        _push(self._with_double_slash_path(encoded_url))
                    else:
                        no_auth_url = urlunsplit((parts.scheme or "rtsp", f"{host}{port}", path, parts.query, parts.fragment))
                        _push(no_auth_url)
                        _push(self._with_double_slash_path(no_auth_url))
            except Exception:
                pass

        return candidates

    def _open_video_capture(self, rtsp_url):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")
        candidates = self._build_rtsp_candidates(rtsp_url)
        if not candidates:
            return None, None

        print(f"🔎 RTSP候选地址数: {len(candidates)}")

        for candidate in candidates:
            # 仅使用 FFmpeg 后端，避免 CAP_ANY 落到 CAP_IMAGES 触发误导性异常日志。
            try:
                print(f"🔁 尝试拉流: {candidate}")
                if candidate == 0:
                    cap = cv2.VideoCapture(0)
                else:
                    cap = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)

                if cap.isOpened():
                    print(f"✅ 拉流候选可用: {candidate}")
                    return cap, candidate

                cap.release()
            except Exception as e:
                print(f"⚠️ VideoCapture 打开失败: {candidate} | {e}")
                continue

        return None, None

    def _monitor_loop(self, device_id, rtsp_url, record_rtsp_url, algo_type_str, stop_event):
        print(f"📷 正在连接视频流: {rtsp_url}")

        # ========= DEBUG 模式 =========
        DEBUG_MODE = os.getenv("AI_DEBUG", "0") == "1"

        if DEBUG_MODE:
            print("🔥 DEBUG模式：四功能并行测试")

            test_algos = list(self.algo_handlers.keys())

            while not stop_event.is_set():
                for algo in test_algos:
                    details = {
                        "type": f"DEBUG-{algo}",
                        "msg": f"{algo} 功能链路测试报警",
                    }
                    self._save_alarm_to_db(device_id, details, "")
                time.sleep(5)

            print(f"--- DEBUG线程已退出: {device_id} ---")
            return

        # ========= 正常视频逻辑 =========
        try:
            cap, used_url = self._open_video_capture(rtsp_url)

            if cap is None:
                print("❌ 视频流打开失败")
                return
            print(f"✅ 视频流连接成功: {used_url}")

            # AI 拉流成功后再启动录像，避免部分设备因并发连接导致 AI 打不开。
            try:
                video_id = int(device_id)
                if record_rtsp_url:
                    self.video_service.start_ffmpeg_recording(video_id, record_rtsp_url)
            except Exception as e:
                print(f"⚠️ 启动分段录像失败(不影响AI检测): {e}")

        except Exception as e:
            print(f"❌ 视频流异常: {e}")
            return

        active_algos = [x.strip() for x in algo_type_str.split(",") if x.strip()]
        frame_interval = 5
        frame_count = 0

        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                time.sleep(2)
                continue

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            try:
                for algo_key in active_algos:

                    if algo_key not in self.algo_handlers:
                        print(f"⚠️ 未识别算法类型: {algo_key}")
                        continue

                    is_alarm, details = self.algo_handlers[algo_key](frame)

                    if is_alarm:
                        img_path = self._save_alarm_image(frame, device_id, details)
                        self._save_alarm_to_db(device_id, details, img_path)

            except Exception as logic_error:
                print(f"⚠️ 逻辑异常: {logic_error}")

            time.sleep(0.02)

        cap.release()
        print(f"--- 监控线程已退出: {device_id} ---")

    # =========================
    # 保存报警图片
    # =========================
    def _save_alarm_image(self, frame, device_id, details=None):
        try:
            # 1. 如果有报警详情，先在图片上绘制报警框
            draw_frame = frame.copy()
            if details and isinstance(details, dict) and "boxes" in details:
                draw_frame = self._draw_boxes_on_frame(draw_frame, details["boxes"])

            # 2. 生成文件名并保存图片
            filename = f"{device_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
            filepath = os.path.join(self.static_dir, filename)

            cv2.imwrite(filepath, draw_frame)
            return f"/static/alarms/{filename}"

        except Exception as e:
            print(f"❌ 图片保存失败: {e}")
            return ""

    def _draw_boxes_on_frame(self, frame, boxes):
        """
        在图片上绘制报警框和中文标注 (解决乱码问题)。
        使用 Pillow 库进行中文绘制。
        """
        try:
            # OpenCV 转 Pillow
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # 加载中文字体 (Windows 默认字体，如果 Linux 需替换路径)
            font_path = r"C:\Windows\Fonts\simhei.ttf"
            if not os.path.exists(font_path):
                # 尝试其他常见中文字体名
                font_path = r"C:\Windows\Fonts\msyh.ttc"
            
            try:
                # 字体大小根据图片高度动态调整
                font_size = max(18, int(frame.shape[0] * 0.03))
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                # 极端情况回退默认
                font = ImageFont.load_default()

            for box in boxes:
                coords = box.get("coords")
                label_type = box.get("type", "异常")
                msg = box.get("msg", "")

                if not coords or len(coords) < 4:
                    continue

                x1, y1, x2, y2 = map(int, coords)

                # 绘制红色报警框 (线宽动态)
                line_width = max(2, int(frame.shape[0] * 0.005))
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_width)

                # 标注信息 (模拟人员名称展示)
                # 备注：实际可关联 face_recognition 的结果
                person_info = "人员: 模拟用户(李工)"
                display_text = f"{label_type}\n{person_info}\n{msg}"

                # 绘制文字背景条
                try:
                    # 使用 textbbox 获取文本尺寸 (Pillow 8.0+)
                    bbox = draw.textbbox((x1, y1 - 10), display_text, font=font)
                    # 往上或往下绘制背景，防止文字出界 (简化始终画在框顶部附近，略带半透明)
                    bg_rect = [bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5]
                    draw.rectangle(bg_rect, fill=(255, 0, 0, 180))
                except Exception:
                    pass

                # 绘制白色文字
                draw.text((x1, y1 - font_size - 10), display_text, font=font, fill=(255, 255, 255))

            # Pillow 转回 OpenCV
            return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

        except Exception as draw_err:
            print(f"⚠️ 图片标注绘制失败: {draw_err}")
            return frame


    def _save_alarm_clip_async(self, alarm_id: int, device_id: str, alarm_time: datetime):
        def _worker():
            try:
                video_id = int(device_id)
            except Exception:
                self._update_alarm_recording_status(alarm_id, "failed", None, "device_id 非摄像头ID，无法自动录像")
                return

            # 等待到“报警后2分钟窗口”结束，且尾部分段达到可拼接成熟期，避免末段仍在写入导致失败。
            mature_buffer = RECORD_SEGMENT_SECONDS + RECORD_SEGMENT_SAFE_MARGIN_SECONDS
            wait_seconds = (alarm_time + timedelta(minutes=2, seconds=mature_buffer) - datetime.now()).total_seconds()
            if wait_seconds > 0:
                time.sleep(min(wait_seconds, 300))

            clip_start = alarm_time - timedelta(minutes=2)
            clip_end = alarm_time + timedelta(minutes=2)

            last_error = None
            for attempt in range(1, 3):
                try:
                    result = self.video_service.save_playback_clip(
                        video_id,
                        clip_start,
                        clip_end,
                        output_type="alarm",
                        filename_prefix=f"alarm_{alarm_id}",
                    )
                    self._update_alarm_recording_status(
                        alarm_id,
                        "saved",
                        result.get("recording_path"),
                        None,
                    )
                    print(f"✅ 报警视频已保存 (alarm_id={alarm_id}): {result.get('recording_path')}")
                    return
                except Exception as e:
                    last_error = e
                    if attempt < 2:
                        time.sleep(max(8, RECORD_SEGMENT_SAFE_MARGIN_SECONDS))

            self._update_alarm_recording_status(alarm_id, "failed", None, str(last_error))
            print(f"❌ 报警视频保存失败 (alarm_id={alarm_id}): {last_error}")

        threading.Thread(target=_worker, daemon=True).start()

    def _update_alarm_recording_status(self, alarm_id: int, status: str, path: str | None, error: str | None):
        db = SessionLocal()
        try:
            record = db.query(AlarmRecord).filter(AlarmRecord.id == alarm_id).first()
            if not record:
                return
            record.recording_status = status
            if path:
                record.recording_path = path
            if error:
                record.recording_error = error[:255]
            db.commit()
        except Exception as e:
            print(f"⚠️ 更新报警录像状态失败 alarm_id={alarm_id}: {e}")
            db.rollback()
        finally:
            db.close()

    # =========================
    # 写数据库
    # =========================
    def _save_alarm_to_db(self, device_id, details, image_path):
        if not details:
            return None

        # 兼容两种返回格式:
        # 1) {"type": "...", "msg": "..."}
        # 2) {"alarm": true, "boxes": [{"type": "...", "msg": "..."}]}
        alarm_type = details.get("type") if isinstance(details, dict) else None
        alarm_msg = details.get("msg") if isinstance(details, dict) else None

        if isinstance(details, dict) and isinstance(details.get("boxes"), list) and details["boxes"]:
            first_box = details["boxes"][0] or {}
            alarm_type = alarm_type or first_box.get("type")
            alarm_msg = alarm_msg or first_box.get("msg")

        if not alarm_type:
            alarm_type = "unknown"
        if not alarm_msg:
            alarm_msg = "检测到异常"

        db = SessionLocal()

        try:
            record = AlarmRecord(
                device_id=str(device_id),
                alarm_type=alarm_type,
                severity="HIGH",
                description=alarm_msg,
                status="pending",
                timestamp=datetime.now(),
                recording_status="pending",
            )

            db.add(record)
            db.commit()
            db.refresh(record)

            self._save_alarm_clip_async(record.id, str(device_id), record.timestamp or datetime.now())

            print(f"✅ 报警已保存 (ID: {record.id})")
            return record.id

        except Exception as e:
            print(f"❌ 数据库写入失败: {e}")
            db.rollback()
            return None
        finally:
            db.close()


ai_manager = AIManager()