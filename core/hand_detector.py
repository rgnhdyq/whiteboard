import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
import time
import sys
import importlib

class MediaPipeAdapter:
    """MediaPipe 版本适配器，自动检测可用API"""
    
    def __init__(self):
        self.version = None
        self.api_type = None  # 'tasks' (新版), 'solutions' (旧版), 'none'
        self.available_modules = {}
        
        self._detect_api()
    
    def _detect_api(self):
        """检测可用的 MediaPipe API"""
        print("[检测] MediaPipe API...")
        
        try:
            import mediapipe as mp
            self.version = getattr(mp, '__version__', 'unknown')
            print(f"  MediaPipe 版本: {self.version}")
            
            # 尝试导入新版 API (tasks)
            try:
                from mediapipe import tasks
                print("  [OK] 找到 tasks 模块 (新版API)")
                self.available_modules['tasks'] = tasks
                
                # 检查 tasks.python.vision 是否可用（正确路径）
                try:
                    from mediapipe.tasks.python import vision
                    print("  [OK] 找到 tasks.python.vision 模块")
                    self.available_modules['vision'] = vision
                    self.api_type = 'tasks'
                    return
                except ImportError:
                    print("  [--] tasks.python.vision 不可用")
            except ImportError:
                print("  [--] tasks 模块不可用")
            
            # 尝试导入旧版 API (solutions)
            try:
                from mediapipe import solutions
                print("  [OK] 找到 solutions 模块 (旧版API)")
                self.available_modules['solutions'] = solutions
                self.api_type = 'solutions'
                return
            except ImportError:
                print("  [--] solutions 模块不可用")
            
            # 检查是否有直接可用的模块
            module_names = [name for name in dir(mp) if not name.startswith('_')]
            print(f"  mediapipe 直接包含: {', '.join(module_names)}")
            
            # 尝试直接查找 hands 模块
            if hasattr(mp, 'hands'):
                print("  [OK] 找到直接可用的 hands 模块")
                self.available_modules['hands'] = mp.hands
                self.api_type = 'direct'
                return
            
            print("  [WARN]️ 无法确定可用的API类型")
            self.api_type = 'none'
            
        except ImportError as e:
            print(f"  [--] 无法导入 mediapipe: {e}")
            self.api_type = 'none'
    
    def create_hand_detector(self, **kwargs):
        """根据检测到的API创建手部检测器"""
        # 参数名映射：max_num_hands -> num_hands（Tasks API 用 num_hands）
        if 'max_num_hands' in kwargs and 'num_hands' not in kwargs:
            kwargs['num_hands'] = kwargs.pop('max_num_hands')
        
        if self.api_type == 'tasks':
            return TasksHandDetector(**kwargs)
        elif self.api_type == 'solutions':
            return SolutionsHandDetector(**kwargs)
        elif self.api_type == 'direct':
            return DirectHandDetector(**kwargs)
        else:
            raise ImportError("没有可用的 MediaPipe API。请确保正确安装 mediapipe: pip install mediapipe")
    
    def is_available(self):
        """检查是否有可用的API"""
        return self.api_type != 'none'


class BaseHandDetector:
    """手部检测器基类"""
    
    def __init__(self):
        # 关键点索引常量
        self.LANDMARK_INDEX = {
            'wrist': 0,
            'thumb_cmc': 1, 'thumb_mcp': 2, 'thumb_ip': 3, 'thumb_tip': 4,
            'index_mcp': 5, 'index_pip': 6, 'index_dip': 7, 'index_tip': 8,
            'middle_mcp': 9, 'middle_pip': 10, 'middle_dip': 11, 'middle_tip': 12,
            'ring_mcp': 13, 'ring_pip': 14, 'ring_dip': 15, 'ring_tip': 16,
            'pinky_mcp': 17, 'pinky_pip': 18, 'pinky_dip': 19, 'pinky_tip': 20
        }
        
        # 性能统计
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
    
    def process_frame(self, frame: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """处理一帧，返回第一只手的关键点（向后兼容）"""
        result = self.process_frame_multi(frame)
        return result[0] if result else None
    
    def process_frame_multi(self, frame: np.ndarray) -> List[List[Tuple[float, float, float]]]:
        """处理一帧，返回所有检测到的手的关键点列表"""
        raise NotImplementedError
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """在图像上绘制关键点和连接线"""
        frame_copy = frame.copy()
        height, width = frame.shape[:2]
        
        if not landmarks:
            return frame_copy
        
        # 绘制关键点
        for i, landmark in enumerate(landmarks):
            x = int(landmark[0] * width)
            y = int(landmark[1] * height)
            
            # 根据关键点类型设置颜色
            if i == 0:  # 手腕
                color = (255, 0, 0)  # 蓝色
                radius = 6
            elif i in [4, 8, 12, 16, 20]:  # 指尖
                color = (0, 255, 0)  # 绿色
                radius = 8
            else:  # 其他关键点
                color = (0, 0, 255)  # 红色
                radius = 4
            
            cv2.circle(frame_copy, (x, y), radius, color, -1)
            cv2.circle(frame_copy, (x, y), radius + 2, (255, 255, 255), 1)
        
        # 绘制连接线
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
            (5, 9), (9, 13), (13, 17)  # 手掌连接
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_x = int(landmarks[start_idx][0] * width)
                start_y = int(landmarks[start_idx][1] * height)
                end_x = int(landmarks[end_idx][0] * width)
                end_y = int(landmarks[end_idx][1] * height)
                
                cv2.line(frame_copy, (start_x, start_y), (end_x, end_y), 
                        (255, 255, 0), 2)  # 青色
        
        return frame_copy
    
    def calculate_distance(self, point1: Tuple[float, float, float], 
                          point2: Tuple[float, float, float]) -> float:
        """计算两个点之间的欧氏距离"""
        return np.sqrt((point1[0] - point2[0])**2 + 
                      (point1[1] - point2[1])**2 + 
                      (point1[2] - point2[2])**2)
    
    def _update_fps(self):
        """更新FPS计算"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def get_fps(self) -> float:
        """获取当前FPS"""
        return self.fps
    
    def release(self):
        """释放资源"""
        pass


class TasksHandDetector(BaseHandDetector):
    """新版 MediaPipe Tasks API 检测器"""
    
    def __init__(self, num_hands: int = 1, **kwargs):
        super().__init__()
        
        import os  # 导入 os 模块
        
        try:
            from mediapipe.tasks.python.vision import hand_landmarker
            from mediapipe.tasks.python.core import base_options
            HandLandmarker = hand_landmarker.HandLandmarker
            HandLandmarkerOptions = hand_landmarker.HandLandmarkerOptions
            RunningMode = hand_landmarker._RunningMode
            from mediapipe import Image, ImageFormat
            
            # 模型文件路径（使用无中文路径）
            model_path = r'C:\mediapipe_models\hand_landmarker.task'
            
            # 如果模型不存在，下载它
            if not os.path.exists(model_path):
                print("[下载] 首次运行，正在下载手部检测模型...")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self._download_model(model_path)
            
            # 创建 base_options
            base_opts = base_options.BaseOptions(model_asset_path=model_path)
            
            # 创建手部关键点检测器
            options = HandLandmarkerOptions(
                base_options=base_opts,
                num_hands=num_hands,
                min_hand_detection_confidence=kwargs.get('min_detection_confidence', 0.5),
                min_hand_presence_confidence=kwargs.get('min_detection_confidence', 0.5),
                min_tracking_confidence=kwargs.get('min_tracking_confidence', 0.5),
                running_mode=RunningMode.VIDEO,
            )
            
            self.landmarker = HandLandmarker.create_from_options(options)
            self.Image = Image
            self.ImageFormat = ImageFormat
            self.running_mode = RunningMode
            
            print("[OK] 使用新版 Tasks API")
            
        except Exception as e:
            print(f"创建 TasksHandDetector 失败: {e}")
            raise
    
    def _download_model(self, dest_path: str):
        """下载 MediaPipe 手部检测模型"""
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        print(f"从 {url} 下载...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"[OK] 模型已保存到: {dest_path}")
        except Exception as e:
            # 如果下载失败，清理并报错
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise RuntimeError(f"下载模型失败: {e}")
    
    def process_frame(self, frame: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """向后兼容，返回第一只手"""
        result = self.process_frame_multi(frame)
        return result[0] if result else None
    
    def process_frame_multi(self, frame: np.ndarray) -> List[List[Tuple[float, float, float]]]:
        """返回所有检测到的手"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self.Image(image_format=self.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.time() * 1000)
            detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            self._update_fps()
            
            hands = []
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                    hands.append(landmarks)
            return hands
        except Exception as e:
            print(f"Tasks API 处理帧时出错: {e}")
            return []
    
    def release(self):
        if hasattr(self, 'landmarker'):
            try:
                self.landmarker.close()
            except:
                pass


class SolutionsHandDetector(BaseHandDetector):
    """旧版 MediaPipe Solutions API 检测器"""
    
    def __init__(self, max_num_hands: int = 1, **kwargs):
        super().__init__()
        
        try:
            from mediapipe.solutions.hands import Hands
            
            self.hands = Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=kwargs.get('min_detection_confidence', 0.5),
                min_tracking_confidence=kwargs.get('min_tracking_confidence', 0.5)
            )
            
            print("[OK] 使用旧版 Solutions API")
            
        except Exception as e:
            print(f"创建 SolutionsHandDetector 失败: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """向后兼容，返回第一只手"""
        result = self.process_frame_multi(frame)
        return result[0] if result else None
    
    def process_frame_multi(self, frame: np.ndarray) -> List[List[Tuple[float, float, float]]]:
        """返回所有检测到的手"""
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.hands.process(image_rgb)
            image_rgb.flags.writeable = True
            self._update_fps()
            
            hands = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    hands.append(landmarks)
            return hands
        except Exception as e:
            print(f"Solutions API 处理帧时出错: {e}")
            return []
    
    def release(self):
        if hasattr(self, 'hands'):
            try:
                self.hands.close()
            except:
                pass


class DirectHandDetector(BaseHandDetector):
    """直接使用 mediapipe.hands 检测器"""
    
    def __init__(self, max_num_hands: int = 1, **kwargs):
        super().__init__()
        
        try:
            import mediapipe as mp
            
            self.hands = mp.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=kwargs.get('min_detection_confidence', 0.5),
                min_tracking_confidence=kwargs.get('min_tracking_confidence', 0.5)
            )
            
            print("[OK] 使用直接 hands API")
            
        except Exception as e:
            print(f"创建 DirectHandDetector 失败: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """向后兼容，返回第一只手"""
        result = self.process_frame_multi(frame)
        return result[0] if result else None
    
    def process_frame_multi(self, frame: np.ndarray) -> List[List[Tuple[float, float, float]]]:
        """返回所有检测到的手"""
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.hands.process(image_rgb)
            image_rgb.flags.writeable = True
            self._update_fps()
            
            hands = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    hands.append(landmarks)
            return hands
        except Exception as e:
            print(f"Direct API 处理帧时出错: {e}")
            return []
    
    def release(self):
        if hasattr(self, 'hands'):
            try:
                self.hands.close()
            except:
                pass


# 主接口类（向后兼容）
class HandDetector:
    """智能手部检测器，自动选择可用API"""
    
    def __init__(self, **kwargs):
        # 创建适配器
        self.adapter = MediaPipeAdapter()
        
        if not self.adapter.is_available():
            print("\n[ERR] 没有可用的 MediaPipe API")
            print("请尝试以下解决方案:")
            print("1. 重新安装 mediapipe: pip install --upgrade mediapipe")
            print("2. 使用 Anaconda 安装: conda install -c conda-forge mediapipe")
            print("3. 降级到兼容版本: pip install mediapipe==0.9.3")
            raise ImportError("无法初始化 MediaPipe 手部检测器")
        
        # 创建实际检测器
        self.detector = self.adapter.create_hand_detector(**kwargs)
        
        # 保存参数用于兼容性
        self.mp_hands = type('mp_hands', (), {})  # 空对象占位
        self.mp_drawing = type('mp_drawing', (), {})
        self.mp_drawing_styles = type('mp_drawing_styles', (), {})
    
    def draw_landmarks_multi(self, frame: np.ndarray, hands: List[List[Tuple[float, float, float]]]) -> np.ndarray:
        """绘制多只手的骨骼，用不同颜色区分"""
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # 绿、蓝、红、青
        for i, hand in enumerate(hands):
            color = colors[i % len(colors)]
            frame = self._draw_single_hand(frame, hand, color)
        # 显示手数量
        cv2.putText(frame, f"Hands: {len(hands)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame
    
    def _draw_single_hand(self, frame: np.ndarray, landmarks: List[Tuple[float, float, float]], 
                          color: Tuple[int, int, int]) -> np.ndarray:
        """绘制单只手的骨骼"""
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        for i, lm in enumerate(landmarks):
            x = int(lm[0] * w)
            y = int(lm[1] * h)
            radius = 8 if i in [4, 8, 12, 16, 20] else 5
            cv2.circle(frame_copy, (x, y), radius, color, -1)
            cv2.circle(frame_copy, (x, y), radius + 1, (255, 255, 255), 1)
        
        connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                       (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                       (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
        for s, e in connections:
            if s < len(landmarks) and e < len(landmarks):
                sx, sy = int(landmarks[s][0]*w), int(landmarks[s][1]*h)
                ex, ey = int(landmarks[e][0]*w), int(landmarks[e][1]*h)
                cv2.line(frame_copy, (sx, sy), (ex, ey), color, 2)
        
        return frame_copy
    
    def process_frame(self, frame: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """处理一帧，返回第一只手（向后兼容）"""
        return self.detector.process_frame(frame)
    
    def process_frame_multi(self, frame: np.ndarray) -> List[List[Tuple[float, float, float]]]:
        """处理一帧，返回所有检测到的手"""
        return self.detector.process_frame_multi(frame)
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        return self.detector.draw_landmarks(frame, landmarks)
    
    def get_landmark_coords(self, landmarks: List[Tuple[float, float, float]], 
                           frame_shape: Tuple[int, int]) -> List[Tuple[int, int, float]]:
        """将归一化坐标转换为像素坐标"""
        height, width = frame_shape[:2]
        coords = []
        for lm in landmarks:
            x = int(lm[0] * width)
            y = int(lm[1] * height)
            z = lm[2]  # 深度信息，保持归一化
            coords.append((x, y, z))
        return coords
    
    def get_specific_landmark(self, landmarks: List[Tuple[float, float, float]], 
                            landmark_name: str) -> Tuple[float, float, float]:
        """获取特定关键点的坐标"""
        idx = self.detector.LANDMARK_INDEX.get(landmark_name)
        if idx is None:
            raise ValueError(f"未知的关键点名称: {landmark_name}")
        
        if idx >= len(landmarks):
            raise IndexError(f"关键点索引 {idx} 超出范围 (共 {len(landmarks)} 个点)")
        
        return landmarks[idx]
    
    def calculate_distance(self, point1: Tuple[float, float, float], 
                          point2: Tuple[float, float, float]) -> float:
        return self.detector.calculate_distance(point1, point2)
    
    def get_fps(self) -> float:
        return self.detector.get_fps()
    
    def release(self):
        self.detector.release()


# 测试函数
def test_all_detectors():
    """测试所有可用的检测器"""
    print("\n[测试] 测试手部检测器...")
    
    # 创建适配器
    adapter = MediaPipeAdapter()
    
    if not adapter.is_available():
        print("没有可用的 MediaPipe API")
        return
    
    print(f"检测到的 API 类型: {adapter.api_type}")
    print(f"MediaPipe 版本: {adapter.version}")
    
    # 测试摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("\n[摄像头] 摄像头测试，按 'q' 退出")
    
    try:
        # 创建检测器
        detector = HandDetector(max_num_hands=1, min_detection_confidence=0.5)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # 检测手部关键点
            landmarks = detector.process_frame(frame)
            
            if landmarks:
                # 绘制关键点
                frame = detector.draw_landmarks(frame, landmarks)
                
                # 显示FPS
                fps = detector.get_fps()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"API: {adapter.api_type}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Points: {len(landmarks)}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Hand Detector Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print("\n[DONE] 测试完成")
        
    except Exception as e:
        print(f"[ERR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_all_detectors()