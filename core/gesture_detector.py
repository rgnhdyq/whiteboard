"""
手势检测器 v4
改动：
  1. 引入 1 Euro Filter 替代简单低通滤波
  2. 引入"空中数位板"：中心 50% 区域绝对映射到全屏
  3. 保留松开缓冲期、防抖、擦除/撤销手势
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import time
from collections import deque


# ==============================================================================
# 1 Euro Filter 实现
# 参考：https://gery.casiez.net/1euro/
# 低速 → 重平滑（去抖），高速 → 轻平滑（低延迟）
# ==============================================================================

class OneEuroFilter:
    """
    1 Euro Filter：自适应低通滤波器。

    参数：
        mincutoff: 最小截止频率（Hz），越小平滑越强。默认 1.0
        beta: 速度系数，越大高速响应越快。默认 0.007
        dcutoff: 导数截止频率（Hz），通常固定 1.0
    """

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.007, dcutoff: float = 1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff

        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        """计算平滑系数 alpha = 1 / (1 + 1/(2*pi*dt*cutoff))"""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt) if dt > 0 else 0.0

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        """
        过滤一个标量值。

        Args:
            x: 原始值
            t: 时间戳（秒）。None 则使用 time.time()
        Returns:
            滤波后的值
        """
        if t is None:
            t = time.time()

        if self._t_prev is None:
            # 第一次调用，直接初始化
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        dt = t - self._t_prev
        if dt <= 0:
            dt = 1.0 / 60.0  # 兜底：假设 60fps

        # 1. 估算变化速度（导数）
        dx = (x - self._x_prev) / dt

        # 2. 对导数做低通滤波
        alpha_d = self._alpha(self.dcutoff, dt)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self._dx_prev

        # 3. 根据速度动态调整截止频率
        #    速度越快 → cutoff 越大 → alpha 越大 → 跟踪越快
        cutoff = self.mincutoff + self.beta * abs(dx_hat)

        # 4. 对值做低通滤波
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1 - alpha) * self._x_prev

        # 更新状态
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t

        return x_hat

    def reset(self):
        """重置滤波器状态（断笔重画时调用，防止飞线）"""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class OneEuroFilter3D:
    """对 (x, y, z) 三元组分别做 1 Euro Filter"""

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.007, dcutoff: float = 1.0):
        self.fx = OneEuroFilter(mincutoff, beta, dcutoff)
        self.fy = OneEuroFilter(mincutoff, beta, dcutoff)
        self.fz = OneEuroFilter(mincutoff, beta, dcutoff)

    def __call__(self, pos: Tuple[float, float, float],
                 t: Optional[float] = None) -> Tuple[float, float, float]:
        return (self.fx(pos[0], t), self.fy(pos[1], t), self.fz(pos[2], t))

    def reset(self):
        self.fx.reset()
        self.fy.reset()
        self.fz.reset()


# ==============================================================================
# 手势检测器
# ==============================================================================

class GestureDetector:

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self.get_default_config()

        self.control_active = True

        # 点击/拖拽
        self.left_click_down = False
        self.last_click_time = 0
        self.left_touch_hold_start = None
        self.is_dragging = False

        # ===== 1 Euro Filter 平滑 =====
        euro_cfg = self.config['smoothing']['euro_filter']
        self._euro_filter = OneEuroFilter3D(
            mincutoff=euro_cfg['mincutoff'],
            beta=euro_cfg['beta'],
            dcutoff=euro_cfg['dcutoff'],
        )
        self._filter_initialized = False

        # ===== 空中数位板：中心区域映射 =====
        tablet_cfg = self.config['tablet']
        self._tablet_x_min = tablet_cfg['x_min']  # 0.25
        self._tablet_x_max = tablet_cfg['x_max']  # 0.75
        self._tablet_y_min = tablet_cfg['y_min']  # 0.25
        self._tablet_y_max = tablet_cfg['y_max']  # 0.75
        self._tablet_x_range = self._tablet_x_max - self._tablet_x_min  # 0.5
        self._tablet_y_range = self._tablet_y_max - self._tablet_y_min  # 0.5

        # 手势防抖
        self._debounce_buffer = {}
        self._debounce_frames = self.config['debounce']['frames']

        # ===== 松开缓冲期 =====
        self.RELEASE_HOLD_FRAMES = 0
        self._release_counter = 0
        self._was_pinching = False

        # ===== 捏合确认窗口（防止误触）=====
        # 捏合必须持续 PINCH_HOLD_SECONDS 秒才开始画
        self.PINCH_HOLD_SECONDS = 1.0
        self._pinch_start_time = 0.0
        self._pinch_confirmed = False  # 捏合已确认（超过1秒）

        # 拍手检测
        self.last_clap_time = 0
        self.clap_cooldown = 2.0
        self.clap_hold_start = None
        self.clap_hold_duration = 0.2
        self.prev_clap_distance = 0

        # 双食指碰撞状态机
        self.index_touching = False
        self.last_index_touches = []

        self.debug = False

    @staticmethod
    def get_default_config() -> Dict:
        return {
            'thresholds': {
                'hand_open': 0.15,
                'palm_radius': 0.08,
                'touch_ratio': 0.6,
            },
            'timing': {
                'double_click_interval': 0.3,
                'touch_confirm_frames': 3,
            },
            'smoothing': {
                # 1 Euro Filter 参数
                'euro_filter': {
                    'mincutoff': 1.0,   # 最小截止频率，越小越平滑
                    'beta': 0.007,      # 速度系数，越大高速响应越快
                    'dcutoff': 1.0,     # 导数截止频率
                },
            },
            # 空中数位板：中心区域映射
            # 摄像头视野中心 50% 区域映射到全屏 0~1
            'tablet': {
                'x_min': 0.25,  # 左边界
                'x_max': 0.75,  # 右边界
                'y_min': 0.25,  # 上边界
                'y_max': 0.75,  # 下边界
            },
            'clap': {'distance_threshold': 0.25},
            'debounce': {'frames': 3},
        }

    # ===================== 防抖 =====================

    def _debounce(self, name: str, raw_value: bool) -> bool:
        if name not in self._debounce_buffer:
            self._debounce_buffer[name] = deque(maxlen=self._debounce_frames)

        buf = self._debounce_buffer[name]
        buf.append(raw_value)

        if len(buf) < self._debounce_frames:
            return False

        return all(buf)

    # ===================== 空中数位板映射 =====================

    def _tablet_map(self, raw_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        将摄像头视野中心区域映射到全屏归一化坐标。

        原始坐标范围：[x_min, x_max] → 映射后：[0.0, 1.0]
        超出中心区域的部分被 clip 截断。
        """
        x = (raw_pos[0] - self._tablet_x_min) / self._tablet_x_range
        y = (raw_pos[1] - self._tablet_y_min) / self._tablet_y_range
        z = raw_pos[2]  # z 不做映射

        # 严格截断到 [0, 1]
        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))

        return (x, y, z)

    # ===================== 核心检测 =====================

    def detect_gestures(self, landmarks_list: List, frame_shape: Tuple[int, int]) -> Dict:
        current_time = time.time()

        hands_data = self._parse_hands(landmarks_list)

        clap_detected = False
        if len(hands_data) >= 2:
            clap_detected = self._detect_index_collision(hands_data[0], hands_data[1], current_time)

        primary_hand = hands_data[0] if hands_data else None

        if primary_hand is None:
            # 手丢失 → 重置滤波器（防止下次出现时飞线）
            if self._filter_initialized:
                self._euro_filter.reset()
                self._filter_initialized = False
            return self._empty_result()

        keypoints = primary_hand

        # 参考指节长度
        index_segment = self._euclidean_distance(
            keypoints['index_mcp'], keypoints.get('index_pip', keypoints['index_mcp']))

        # 手指伸出状态（食指采用放宽的 0.45 门槛，其余手指维持 0.6）
        index_extended = self._is_finger_extended(keypoints, 'index', threshold=0.45)
        thumb_extended = self._is_finger_extended(keypoints, 'thumb')
        middle_extended = self._is_finger_extended(keypoints, 'middle')
        ring_extended = self._is_finger_extended(keypoints, 'ring')
        pinky_extended = self._is_finger_extended(keypoints, 'pinky')

        # 指尖触碰
        thumb_index_touching = self._are_fingers_touching(
            keypoints, 'thumb', 'index', index_segment)
        thumb_ring_touching = False
        if thumb_extended:
            thumb_ring_touching = self._are_fingers_touching(
                keypoints, 'thumb', 'ring', index_segment)

        hand_open = self._detect_hand_open(keypoints)

        # ===== 食指位置处理流水线 =====
        # 1. 原始坐标（归一化 0~1，摄像头视野）
        raw_pos = keypoints['index_tip']

        # 2. 空中数位板映射：中心 50% → 全屏 0~1
        mapped_pos = self._tablet_map(raw_pos)

        # 3. 1 Euro Filter 自适应平滑
        index_pos = self._euro_filter(mapped_pos, current_time)
        self._filter_initialized = True

        # ===== 松开缓冲期 + 捏合确认窗口 =====
        is_pinching = index_extended and self._is_thumb_near_index_pip(keypoints, index_segment)
        drawing_active = False

        if is_pinching:
            if not self._was_pinching:
                # 新的捏合开始，记录起始时间
                self._pinch_start_time = current_time
                self._pinch_confirmed = False
            self._was_pinching = True

            hold_duration = current_time - self._pinch_start_time
            if hold_duration >= self.PINCH_HOLD_SECONDS:
                # 捏合已确认（超过1秒），开始画画
                self._pinch_confirmed = True
                drawing_active = True
                self._release_counter = self.RELEASE_HOLD_FRAMES
            else:
                # 还在确认窗口内，不算画画（避免误触）
                drawing_active = False
            # 进度 0~1
            pinch_progress = min(hold_duration / self.PINCH_HOLD_SECONDS, 1.0)
        elif self._was_pinching:
            if self._pinch_confirmed:
                # 已确认的捏合松开 → 进入缓冲期
                pinch_progress = 1.0
                if self._release_counter > 0:
                    drawing_active = True
                    self._release_counter -= 1
                else:
                    # 缓冲期结束
                    self._was_pinching = False
                    self._pinch_confirmed = False
            else:
                # 未确认的捏合松开（不到1秒就松了）→ 重置，不算画画
                self._was_pinching = False
                self._pinch_confirmed = False
                pinch_progress = 0.0
        else:
            self._was_pinching = False
            self._release_counter = 0
            self._pinch_confirmed = False
            pinch_progress = 0.0

        num_extended = sum([index_extended, middle_extended, ring_extended, pinky_extended])

        # 激光笔
        laser_active = index_extended and not drawing_active

        # 擦除手势：五指张开（放宽条件：只要求食/中/无/小四指伸直，且不与拇指捏合，避免MediaPipe拇指判定不稳造成的擦除中断）
        erase_gesture = (index_extended and middle_extended and ring_extended
                         and pinky_extended and not thumb_index_touching)

        # 模式切换已移至拍手，蜘蛛侠手势废弃
        mode_switch_gesture = False

        # 点击事件
        click_events = self._process_click_events(
            thumb_index_touching, thumb_ring_touching, current_time)

        return {
            'control_active': self.control_active,
            'hand_open': hand_open,
            'index_extended': index_extended,
            'thumb_extended': thumb_extended,
            'thumb_index_touching': thumb_index_touching,
            'thumb_ring_touching': thumb_ring_touching,
            'index_position': index_pos,
            'raw_position': raw_pos,        # 原始坐标（调试用）
            'mapped_position': mapped_pos,   # 数位板映射后（调试用）
            'click_events': click_events,
            'keypoints': keypoints,
            'middle_extended': middle_extended,
            'ring_extended': ring_extended,
            'pinky_extended': pinky_extended,
            # 手势状态
            'laser_active': laser_active,
            'drawing_active': drawing_active,
            'pinch_progress': pinch_progress,
            'erase_gesture': erase_gesture,
            'mode_switch_gesture': mode_switch_gesture,
            # 动态手势
            'clap_detected': clap_detected,
            'num_hands': len(hands_data),
        }

    # ===================== 点击事件 =====================

    def _process_click_events(self, thumb_index_touching: bool,
                               thumb_ring_touching: bool,
                               current_time: float) -> Dict:
        click_events = {
            'left_click': False, 'right_click': False,
            'double_click': False, 'drag_start': False, 'drag_end': False,
        }

        if thumb_index_touching:
            if not self.left_touch_hold_start:
                self.left_touch_hold_start = current_time
                click_events['left_click'] = True
            else:
                hold_duration = current_time - self.left_touch_hold_start
                if hold_duration > self.config['timing']['touch_confirm_frames'] * 0.1:
                    if not self.is_dragging:
                        click_events['drag_start'] = True
                        self.is_dragging = True
        else:
            if self.left_touch_hold_start:
                self.left_touch_hold_start = None
                if self.is_dragging:
                    click_events['drag_end'] = True
                    self.is_dragging = False

        if thumb_ring_touching:
            click_events['right_click'] = True

        return click_events

    # ===================== 双手解析 =====================

    def _parse_hands(self, landmarks_list) -> List[Dict]:
        if landmarks_list is None:
            return []

        if isinstance(landmarks_list, list) and len(landmarks_list) > 0:
            if isinstance(landmarks_list[0], list) and len(landmarks_list[0]) >= 21:
                return [self._extract_keypoints(hand) for hand in landmarks_list[:2]]
            elif isinstance(landmarks_list[0], tuple) and len(landmarks_list[0]) == 3:
                return [self._extract_keypoints(landmarks_list)]
            elif hasattr(landmarks_list[0], 'landmark'):
                return [self._extract_keypoints(landmarks_list[0])]

        if hasattr(landmarks_list, 'landmark'):
            return [self._extract_keypoints(landmarks_list)]

        return []

    # ===================== 关键点提取 =====================

    def _extract_keypoints(self, landmarks) -> Dict[str, Tuple[float, float, float]]:
        if isinstance(landmarks, list) and len(landmarks) >= 21:
            return {
                'wrist': landmarks[0],
                'thumb_cmc': landmarks[1], 'thumb_mcp': landmarks[2],
                'thumb_ip': landmarks[3], 'thumb_tip': landmarks[4],
                'index_mcp': landmarks[5], 'index_pip': landmarks[6],
                'index_dip': landmarks[7], 'index_tip': landmarks[8],
                'middle_mcp': landmarks[9], 'middle_pip': landmarks[10],
                'middle_dip': landmarks[11], 'middle_tip': landmarks[12],
                'ring_mcp': landmarks[13], 'ring_pip': landmarks[14],
                'ring_dip': landmarks[15], 'ring_tip': landmarks[16],
                'pinky_mcp': landmarks[17], 'pinky_pip': landmarks[18],
                'pinky_dip': landmarks[19], 'pinky_tip': landmarks[20],
            }
        elif hasattr(landmarks, 'landmark'):
            lm = landmarks.landmark
            return {
                'wrist': (lm[0].x, lm[0].y, lm[0].z),
                'thumb_cmc': (lm[1].x, lm[1].y, lm[1].z),
                'thumb_mcp': (lm[2].x, lm[2].y, lm[2].z),
                'thumb_ip': (lm[3].x, lm[3].y, lm[3].z),
                'thumb_tip': (lm[4].x, lm[4].y, lm[4].z),
                'index_mcp': (lm[5].x, lm[5].y, lm[5].z),
                'index_pip': (lm[6].x, lm[6].y, lm[6].z),
                'index_dip': (lm[7].x, lm[7].y, lm[7].z),
                'index_tip': (lm[8].x, lm[8].y, lm[8].z),
                'middle_mcp': (lm[9].x, lm[9].y, lm[9].z),
                'middle_pip': (lm[10].x, lm[10].y, lm[10].z),
                'middle_dip': (lm[11].x, lm[11].y, lm[11].z),
                'middle_tip': (lm[12].x, lm[12].y, lm[12].z),
                'ring_mcp': (lm[13].x, lm[13].y, lm[13].z),
                'ring_pip': (lm[14].x, lm[14].y, lm[14].z),
                'ring_dip': (lm[15].x, lm[15].y, lm[15].z),
                'ring_tip': (lm[16].x, lm[16].y, lm[16].z),
                'pinky_mcp': (lm[17].x, lm[17].y, lm[17].z),
                'pinky_pip': (lm[18].x, lm[18].y, lm[18].z),
                'pinky_dip': (lm[19].x, lm[19].y, lm[19].z),
                'pinky_tip': (lm[20].x, lm[20].y, lm[20].z),
            }
        else:
            raise ValueError(f"未知的 landmarks 格式: {type(landmarks)}")

    # ===================== 手指判断 =====================

    def _is_finger_extended(self, keypoints: Dict, finger: str, threshold: float = 0.6) -> bool:
        wrist = keypoints['wrist']
        tip = keypoints.get(finger + '_tip')
        mcp = keypoints.get(finger + '_mcp')
        if not tip or not mcp:
            return False

        tip_mcp = self._euclidean_distance(tip, mcp)
        mcp_wrist = self._euclidean_distance(mcp, wrist)

        if mcp_wrist < 0.01:
            return False

        ratio = tip_mcp / mcp_wrist

        # 针对食指的透视缩短（正指屏幕）进行智能生理排他性补偿
        if finger == 'index' and threshold <= 0.45:
            if ratio > threshold:
                return True
            if ratio > 0.05:
                # 检查中指、无名指、小指是否都处于弯曲蜷缩状态
                other_curled = True
                for f in ['middle', 'ring', 'pinky']:
                     f_tip = keypoints.get(f + '_tip')
                     f_mcp = keypoints.get(f + '_mcp')
                     if f_tip and f_mcp:
                         f_ratio = self._euclidean_distance(f_tip, f_mcp) / mcp_wrist
                         if f_ratio > 0.45:
                             other_curled = False
                             break
                if other_curled:
                    return True

        return ratio > threshold

    def _is_thumb_near_index_pip(self, keypoints: Dict, reference_length: float = None) -> bool:
        thumb_tip = keypoints.get('thumb_tip')
        index_pip = keypoints.get('index_pip')
        index_tip = keypoints.get('index_tip')
        wrist = keypoints.get('wrist')
        index_mcp = keypoints.get('index_mcp')
        if not all([thumb_tip, index_pip, index_tip, wrist, index_mcp]):
            return False
            
        # 同时计算大拇指尖到食指第二关节及指尖的 2D 二维投影距离，取最小值
        d_thumb_pip_2d = np.sqrt((thumb_tip[0] - index_pip[0]) ** 2 +
                                 (thumb_tip[1] - index_pip[1]) ** 2)
        d_thumb_tip_2d = np.sqrt((thumb_tip[0] - index_tip[0]) ** 2 +
                                 (thumb_tip[1] - index_tip[1]) ** 2)
        min_d_2d = min(d_thumb_pip_2d, d_thumb_tip_2d)

        d_palm_2d = np.sqrt((index_mcp[0] - wrist[0]) ** 2 +
                            (index_mcp[1] - wrist[1]) ** 2)
        
        if d_palm_2d < 0.01:
            return False

        # 适当放宽比例阈值至 0.45，包容微弯和指向屏幕时的透视压缩
        return min_d_2d < d_palm_2d * 0.45

    def _are_fingers_touching(self, keypoints: Dict, finger1: str, finger2: str,
                               reference_length: float = None) -> bool:
        tip1 = keypoints.get(finger1 + '_tip')
        tip2 = keypoints.get(finger2 + '_tip')
        if not tip1 or not tip2:
            return False

        distance = self._euclidean_distance(tip1, tip2)

        if reference_length and reference_length > 0.01:
            return distance < reference_length * self.config['thresholds']['touch_ratio']

        return distance < 0.05

    def _detect_hand_open(self, keypoints: Dict) -> bool:
        wrist = keypoints['wrist']
        fingertips = [
            keypoints['thumb_tip'], keypoints['index_tip'],
            keypoints['middle_tip'], keypoints['ring_tip'], keypoints['pinky_tip'],
        ]
        distances = [self._euclidean_distance(wrist, tip) for tip in fingertips]
        avg_distance = np.mean(distances)
        normalized_distance = avg_distance / self.config['thresholds']['palm_radius']
        return normalized_distance > self.config['thresholds']['hand_open']



    # ===================== 拍手 =====================

    def _detect_index_collision(self, hand1: Dict, hand2: Dict, current_time: float) -> bool:
        # 1. 检查两只手的食指是否都伸直
        hand1_index_extended = self._is_finger_extended(hand1, 'index', threshold=0.45)
        hand2_index_extended = self._is_finger_extended(hand2, 'index', threshold=0.45)
        
        if not (hand1_index_extended and hand2_index_extended):
            # 如果有一只手的食指没伸出，视为分开状态
            self.index_touching = False
            return False

        # 2. 计算食指指尖之间的 3D 欧氏距离
        d_tips = self._euclidean_distance(hand1['index_tip'], hand2['index_tip'])
        
        # 归一化参考长度（使用 hand1 的手掌长度）
        palm_len = self._euclidean_distance(hand1['wrist'], hand1['index_mcp'])
        if palm_len < 0.01:
            palm_len = 0.12  # 兜底值
            
        touch_threshold = palm_len * 0.45  # 大约 0.05
        release_threshold = palm_len * 0.65  # 大约 0.08
        
        if d_tips < touch_threshold:
            if not self.index_touching:
                # 刚刚产生接触，记录时间戳
                self.index_touching = True
                self.last_index_touches.append(current_time)
                # 只保留最近两次接触时间戳
                if len(self.last_index_touches) > 2:
                    self.last_index_touches.pop(0)
                
                # 后台实时打印碰撞检测日志与时间戳
                if len(self.last_index_touches) == 1:
                    print(f"[手势] 第一次食指相碰检测成功！时间戳: {current_time:.2f}")
                elif len(self.last_index_touches) == 2:
                    print(f"[手势] 第二次食指相碰检测成功！时间戳: {current_time:.2f}")
                
                # 检查是否满足 2 秒内碰撞两次
                if len(self.last_index_touches) == 2:
                    t1, t2 = self.last_index_touches
                    if t2 - t1 <= 2.0:
                        # 触发双击碰撞，重置历史防连续触发
                        self.last_index_touches.clear()
                        print(f"[手势] 2秒内食指碰撞两次触发模式切换！时间差: {t2 - t1:.2f}秒")
                        return True
        elif d_tips > release_threshold:
            # 已经拉开距离，重置接触状态，允许下一次碰撞产生
            self.index_touching = False
            
        return False

    # ===================== 工具 =====================

    def _empty_result(self) -> Dict:
        return {
            'control_active': self.control_active,
            'hand_open': False, 'index_extended': False, 'thumb_extended': False,
            'thumb_index_touching': False, 'thumb_ring_touching': False,
            'index_position': (0, 0, 0),
            'raw_position': (0, 0, 0),
            'mapped_position': (0, 0, 0),
            'click_events': {'left_click': False, 'right_click': False,
                             'double_click': False, 'drag_start': False, 'drag_end': False},
            'keypoints': {},
            'middle_extended': False, 'ring_extended': False, 'pinky_extended': False,
            'laser_active': False, 'drawing_active': False,
            'pinch_progress': 0.0,
            'erase_gesture': False, 'mode_switch_gesture': False,
            'clap_detected': False, 'num_hands': 0,
        }

    @staticmethod
    def _euclidean_distance(p1, p2) -> float:
        return np.sqrt((p1[0] - p2[0]) ** 2 +
                       (p1[1] - p2[1]) ** 2 +
                       (p1[2] - p2[2]) ** 2)

    def reset_state(self):
        self.control_active = True
        self.left_click_down = False
        self.last_click_time = 0
        self.left_touch_hold_start = None
        self.is_dragging = False
        self.clap_hold_start = None
        self.prev_clap_distance = 0
        self._debounce_buffer.clear()
        self._release_counter = 0
        self._was_pinching = False
        self._pinch_start_time = 0.0
        self._pinch_confirmed = False
        # 重置 1 Euro Filter
        self._euro_filter.reset()
        self._filter_initialized = False
