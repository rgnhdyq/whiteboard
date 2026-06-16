"""
PyQt5 透明全屏覆盖窗口
直接在屏幕上绘制，无需截图
"""

import sys
import time
import queue
import numpy as np
from typing import Optional, Tuple, List
from collections import deque

from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage, QFont, QBrush, QPainterPath, QPolygon, QPolygonF, QGuiApplication, QRegion

from core.floating_menu import FloatingMenu, COLORS as MENU_COLORS, THICKNESSES as MENU_THICKNESSES


class OverlayWindow(QWidget):
    """透明全屏覆盖窗口 — 手势画布"""

    # 画笔颜色（与 FloatingMenu 同源）
    COLORS = [c for c, _ in MENU_COLORS]
    COLOR_NAMES = [n for _, n in MENU_COLORS]

    def __init__(self, gesture_queue: queue.Queue, engine=None):
        super().__init__()
        self.gesture_queue = gesture_queue
        self.engine = engine

        # 窗口设置：无边框 + 置顶 + 透明 + 鼠标穿透（允许鼠标操作电脑）
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.WindowTransparentForInput |
            Qt.Tool  # 不显示在任务栏
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 全屏
        screen = QApplication.primaryScreen().geometry()
        self.screen_w = screen.width()
        self.screen_h = screen.height()
        self.setGeometry(screen)

        # 绘画状态
        self.strokes: List[List[QPoint]] = []  # 所有笔画
        self.current_stroke: List[QPoint] = []  # 当前笔画
        self.is_drawing = False
        self.is_erasing = False  # 擦除状态（独立追踪）

        # 画笔设置
        self.pen_color = self.COLORS[0]
        self.pen_thickness = 4
        self.eraser_thickness = 60
        self.is_eraser = False
        self._color_index = 0

        # 擦除位置（保留最后位置，手短暂丢失不清除）
        self.erase_pos: Optional[QPoint] = None

        # 激光笔及拖影特效
        self.laser_pos: Optional[QPoint] = None
        self.laser_trail = deque(maxlen=25) # 记录最近25帧的轨迹作为拖影

        # 撤销历史（保存笔画快照）
        self.history: List[List[List[QPoint]]] = []
        self.redo_stack: List[List[List[QPoint]]] = []
        self.max_history = 20

        # 摄像头预览
        self.camera_image: Optional[QImage] = None

        # 模式
        self.mode = "whiteboard"  # "whiteboard" / "annotation"

        # 五指擦除状态
        self.five_finger_eraser = False
        self._last_erase_state = False

        # 捏合确认进度（0~1，用于视觉反馈）
        self._pinch_progress = 0.0

        # 状态文字
        self.status_text = "就绪"
        self.fps_text = "FPS: 0"

        # 选区缩放状态管理
        self.selected_region: Optional[dict] = None
        self.region_state = "none" # none, drawn_wait, selected, dragging
        self.region_hover_frames = 0
        self.HOVER_SELECT_FRAMES = 90  # ~1.5s
        self._last_two_hand_dist = 0
        self.dynamic_feedback_text = ""
        self.dynamic_feedback_timer = 0

        # LLM 分析结果（右上角白框显示）
        self._llm_result_text = ""
        self._llm_result_active = False

        # 重新拿起已放置选区的状态
        self._pickup_hover_frames = 0
        self._pickup_target_index = None
        self._pickup_cooldown_until = 0.0  # 冷却期：放下后不立刻拾取

        # 悬浮菜单（嵌入式，画在 OverlayWindow 内部）
        self.floating_menu = FloatingMenu(screen_w=self.screen_w, screen_h=self.screen_h)
        self._finger_pos = QPoint(0, 0)  # 当前食指屏幕坐标

        # 定时器：持续拉取后台检测结果 (约 120fps 极速轮询)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_gesture)
        self.poll_timer.start(8)  # ~120fps 保证绝对不漏掉后台线程的任何一帧

        # 定时器：更新摄像头预览
        self.cam_timer = QTimer(self)
        self.cam_timer.timeout.connect(self._update_camera)
        self.cam_timer.start(16)  # ~60fps (最高尝试以60帧刷新右上角预览)

        # 键盘快捷键
        self.grabKeyboard()

        print(f"[Overlay] 窗口创建: {self.screen_w}x{self.screen_h}")

    def _set_selected_region(self, region):
        """设置选中区域"""
        self.selected_region = region

    # ===================== 手势轮询 =====================

    def _poll_gesture(self):
        """从队列读取手势数据并处理"""
        # 动态手势开关：
        #   - 有选区且非拖动状态 → 开启
        #   - 拖动中 / 无选区 / 非批注模式 → 关闭
        region_dragging = (self.selected_region is not None and self.region_state == "dragging")
        enable_dynamic = (self.mode == "annotation" and self.selected_region is not None and not region_dragging)
        if getattr(self, '_dynamic_enabled_state', None) != enable_dynamic:
            self._dynamic_enabled_state = enable_dynamic
            print(f"[Overlay] enable_dynamic = {enable_dynamic} (region={self.selected_region is not None}, state={self.region_state})")
            if self.engine:
                self.engine.set_dynamic_gesture_enabled(enable_dynamic)

        try:
            gesture_data = self.gesture_queue.get_nowait()
        except queue.Empty:
            return

        self._process_gesture(gesture_data)
        
        # 智能按需刷新：若有手在画面中，或者菜单在缩回/展开/悬停等动画中，才重绘全屏，大幅节省 CPU
        need_update = False
        if self.laser_pos is not None:
            need_update = True
        elif self.floating_menu.is_animating():
            need_update = True

        if need_update:
            self.update()

    def _process_gesture(self, data: dict):
        """处理手势数据"""
        index_pos = data.get('index_position')
        drawing = data.get('drawing_active', False)
        pinch_progress = data.get('pinch_progress', 0.0)
        erase = data.get('erase_gesture', False)
        clap = data.get('clap_detected', False)
        mode_switch = data.get('mode_switch_gesture', False)
        num_hands = data.get('num_hands', 0)
        index_extended = data.get('index_extended', False)

        self._pinch_progress = pinch_progress

        # 仅在擦除状态转换时打印一次
        if erase and not self._last_erase_state:
            print("[手势] 进入五指擦除模式")
        elif not erase and self._last_erase_state:
            print("[手势] 退出五指擦除模式")
        self._last_erase_state = erase

        if clap:
            print("[手势] 双食指碰撞两次 → 切换模式")

        primary_is_pinching = data.get('primary_is_pinching', False)
        secondary_is_pinching = data.get('secondary_is_pinching', False)
        secondary_pos = data.get('secondary_index_position')
        dynamic_gesture = data.get('dynamic_gesture')

        # ===== 动态大招处理 =====
        if dynamic_gesture:
            print(f"[Overlay] 收到动态手势: {dynamic_gesture} (region={self.selected_region is not None}, state={self.region_state})")
        # 拖动中不响应动态手势（enable_dynamic 可能有延迟，这里做二次守卫）
        if dynamic_gesture and dynamic_gesture in ("V", "X") and self.region_state != "dragging":
            print(f"[手势] 触发动态大招: {dynamic_gesture}")

            if dynamic_gesture == "X":
                if self.selected_region is not None:
                    self.dynamic_feedback_text = "X: 已擦除"
                    self.dynamic_feedback_timer = 120
                    # 彻底擦除：不放回画布，直接丢弃
                    self._set_selected_region(None)
                    self.region_state = "none"
                    self.region_hover_frames = 0
                    self._llm_result_active = False
                    self._llm_result_text = ""
                    self._pickup_cooldown_until = time.time() + 1.0
                else:
                    self.dynamic_feedback_text = "X: 无选区"
                    self.dynamic_feedback_timer = 60

            elif dynamic_gesture == "V":
                if self.selected_region is not None:
                    self.dynamic_feedback_text = "V: 正在分析..."
                    self.dynamic_feedback_timer = 120
                    self._trigger_llm_analysis()
                else:
                    self.dynamic_feedback_text = "V: 请先框选区域"
                    self.dynamic_feedback_timer = 60

            # 先屏蔽向左挥和向右挥的响应
            # elif dynamic_gesture == "SwipeLeft":
            #     self.dynamic_feedback_text = "⏪ Swipe Left: Undo!"
            #     self.dynamic_feedback_timer = 60
            #     self._undo()
            #
            # elif dynamic_gesture == "SwipeRight":
            #     self.dynamic_feedback_text = "⏩ Swipe Right: Redo!"
            #     self.dynamic_feedback_timer = 60
            #     self._redo()

            # 暂时屏蔽画圈响应
            # elif dynamic_gesture == "Circle":
            #     self.dynamic_feedback_text = "⭕ Circle: Select Region!"
            #     self.dynamic_feedback_timer = 120
            #     # TODO: 开启截屏选区逻辑

        # 没检测到手
        if index_pos is None or index_pos == (0, 0, 0):
            self._stop_stroke()
            self.laser_pos = None
            self.five_finger_eraser = False
            self.status_text = "无手"
            return

        # 屏幕坐标
        x = int(index_pos[0] * self.screen_w)
        y = int(index_pos[1] * self.screen_h)
        x = max(0, min(self.screen_w - 1, x))
        y = max(0, min(self.screen_h - 1, y))

        self.laser_pos = QPoint(x, y)
        self._finger_pos = QPoint(x, y)
        self.five_finger_eraser = erase
        
        # 记录拖影（仅当食指伸直且未捏合时，即“激光笔”状态）
        if index_extended and not primary_is_pinching:
            self.laser_trail.append(QPoint(x, y))
        else:
            self.laser_trail.clear()

        # 区域选择交互逻辑
        region_handled = False
        if self.selected_region:
            polygon = self.selected_region['polygon']
            center = self.selected_region['center']
            is_inside = polygon.containsPoint(QPoint(x, y), Qt.OddEvenFill)

            if self.region_state == "drawn_wait":
                if not primary_is_pinching:
                    if is_inside:
                        self.region_hover_frames += 1
                        if self.region_hover_frames >= self.HOVER_SELECT_FRAMES:
                            self.region_state = "selected"
                            self.selected_region['scale'] = 1.05
                            self.region_hover_frames = 0
                            print("[Overlay] 区域已激活选中")
                    else:
                        # 缓慢衰减进度，防止微小手抖导致瞬间清零
                        self.region_hover_frames = max(0, self.region_hover_frames - 2)
                else:
                    if not is_inside:
                        self._drop_region()

            elif self.region_state == "selected":
                if primary_is_pinching and secondary_is_pinching and secondary_pos:
                    sec_x = int(secondary_pos[0] * self.screen_w)
                    sec_y = int(secondary_pos[1] * self.screen_h)
                    dist = np.sqrt((x - sec_x)**2 + (y - sec_y)**2)
                    if self._last_two_hand_dist > 0:
                        delta = dist - self._last_two_hand_dist
                        scale_change = delta / 500.0
                        self.selected_region['scale'] = max(0.5, min(5.0, self.selected_region['scale'] + scale_change))
                    self._last_two_hand_dist = dist
                    region_handled = True
                else:
                    self._last_two_hand_dist = 0
                    if not primary_is_pinching:
                        if is_inside:
                            self.region_hover_frames += 1
                            if self.region_hover_frames >= 45:
                                self.region_state = "dragging"
                                self.region_hover_frames = 0
                                print("[Overlay] 区域抓取中")
                        else:
                            self.region_hover_frames = max(0, self.region_hover_frames - 2)
                    else:
                        if not is_inside:
                            self._drop_region()

            elif self.region_state == "dragging":
                if primary_is_pinching:
                    print("[Overlay] 区域已放置")
                    self._drop_region()
                else:
                    dx = x - center.x()
                    dy = y - center.y()
                    self.selected_region['center'] = QPoint(x, y)
                    self.selected_region['polygon'].translate(dx, dy)
                    self.selected_region['bbox'].translate(dx, dy)
                    region_handled = True

        if region_handled:
            self._stop_stroke()
            self.status_text = "区域缩放/拖动中"
            return

        # ===== 重新拿起已放置的选区 =====
        if self.selected_region is None and not primary_is_pinching and time.time() > self._pickup_cooldown_until:
            pickup_target = None
            for i, stroke in enumerate(self.strokes):
                if stroke.get('type') == 'pixmap':
                    center = stroke['center']
                    pixmap = stroke['pixmap']
                    # 扩大 hit 区域：pixmap 半径 + 30px 容差
                    hit_radius = max(pixmap.width(), pixmap.height()) // 2 + 30
                    dist_sq = (x - center.x()) ** 2 + (y - center.y()) ** 2
                    if dist_sq < hit_radius ** 2:
                        pickup_target = i
                        break

            if pickup_target is not None:
                if getattr(self, '_pickup_target_index', None) == pickup_target:
                    self._pickup_hover_frames += 1
                    if self._pickup_hover_frames >= 30:  # ~0.5s 悬停后拿起
                        stroke = self.strokes.pop(pickup_target)
                        pixmap = stroke['pixmap']
                        center = stroke['center']
                        pw, ph = pixmap.width(), pixmap.height()
                        bbox = QRect(center.x() - pw // 2, center.y() - ph // 2, pw, ph)
                        # 用 bbox 四角创建 polygon，用于后续 hit test
                        polygon = QPolygon([
                            QPoint(bbox.left(), bbox.top()),
                            QPoint(bbox.right(), bbox.top()),
                            QPoint(bbox.right(), bbox.bottom()),
                            QPoint(bbox.left(), bbox.bottom()),
                        ])
                        self._set_selected_region({
                            'pixmap': pixmap,
                            'polygon': polygon,
                            'bbox': bbox,
                            'center': QPoint(center),
                            'scale': stroke.get('scale', 1.0)
                        })
                        self.region_state = "selected"
                        self.region_hover_frames = 0
                        self._pickup_hover_frames = 0
                        self._pickup_target_index = None
                        print(f"[Overlay] 重新拿起已放置的选区 (index={pickup_target})")
                else:
                    self._pickup_target_index = pickup_target
                    self._pickup_hover_frames = 1
            else:
                self._pickup_target_index = None
                self._pickup_hover_frames = 0
        else:
            self._pickup_target_index = None
            self._pickup_hover_frames = 0

        # 悬浮菜单 hit_test（菜单区域内不执行画画/擦除）
        if self.floating_menu.in_menu_area(x):
            self.floating_menu.hit_test(
                x, y,
                on_color=self._set_color,
                on_thickness=self._set_thickness,
                on_action=self._on_menu_action,
            )
            self._stop_stroke()
            self.status_text = "菜单"
            return
        else:
            self.floating_menu.hit_test(x, y)

        # 五指张开 → 擦除
        if erase:
            self._erase(QPoint(x, y))
            self.status_text = "擦除中"
            return
        else:
            self.is_erasing = False

        # 捏合（含缓冲期）→ 画画
        if drawing:
            if not self.is_drawing:
                self._start_stroke()
            self._draw(QPoint(x, y))
            self.status_text = "画画中" if not self.is_eraser else "橡皮中"
        else:
            self._stop_stroke()
            self.status_text = "激光笔"

        # (双手相碰拍手的模式切换已按要求移除)

        if 'fps' in data:
            self.fps_text = f"FPS: {data['fps']:.0f}"

    # ===================== 绘画控制 =====================

    def _copy_stroke(self, s: dict) -> dict:
        if s.get('type') == 'pixmap':
            return {
                'type': 'pixmap',
                'pixmap': s['pixmap'],
                'center': QPoint(s['center']),
                'scale': s['scale']
            }
        else:
            return {
                'points': s['points'][:],
                'color': s['color'],
                'thickness': s['thickness'],
                'is_eraser': s['is_eraser']
            }

    def _start_stroke(self):
        self.is_drawing = True
        self.current_stroke = []
        self.history.append([self._copy_stroke(s) for s in self.strokes])
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _draw(self, pos: QPoint):
        self.current_stroke.append(pos)

    def _stop_stroke(self):
        if self.is_drawing and self.current_stroke:
            points = self.current_stroke
            is_closed = False
            loop_points = []
            
            # 检测自交叉或闭合回路 (封闭图形)
            # 允许用户画带有长尾巴的圈，或者螺旋线，系统会自动提取出交叉形成的闭环
            if self.mode == "annotation" and len(points) > 15:
                best_i, best_j = -1, -1
                min_dist = 200 ** 2  # 容忍的最大缺口距离（平方）
                
                # 寻找距离最近的两个点构成闭环
                for i in range(len(points) - 15):
                    # 采样优化性能
                    if i % 3 != 0 and i != 0: continue
                    p1 = points[i]
                    for j in range(i + 15, len(points)):
                        if j % 3 != 0 and j != len(points) - 1: continue
                        p2 = points[j]
                        sq_dist = (p1.x() - p2.x())**2 + (p1.y() - p2.y())**2
                        
                        if sq_dist < min_dist:
                            # 验证这个环是否足够大，过滤掉原地打转的细小手抖交叉
                            ring = points[i:j]
                            xs = [p.x() for p in ring]
                            ys = [p.y() for p in ring]
                            w = max(xs) - min(xs)
                            h = max(ys) - min(ys)
                            if w > 30 and h > 30:
                                min_dist = sq_dist
                                best_i = i
                                best_j = j
                
                if best_i != -1:
                    is_closed = True
                    loop_points = points[best_i:best_j+1]

            stroke_obj = {
                'points': points,
                'color': self.pen_color,
                'thickness': self.pen_thickness,
                'is_eraser': self.is_eraser
            }
            
            if is_closed and not self.is_eraser:
                # 使用纯净的闭合回路建立遮罩，过滤掉多余的线段（如长尾巴）
                effective_points = loop_points if loop_points else points
                polygon = QPolygon(effective_points)
                bbox = polygon.boundingRect()
                
                if bbox.width() > 10 and bbox.height() > 10:
                    # 获取屏幕截图并建立遮罩
                    screen = QGuiApplication.primaryScreen()
                    bbox.adjust(-5, -5, 5, 5)
                    bbox = bbox.intersected(QRect(0, 0, self.screen_w, self.screen_h))
                    pixmap = screen.grabWindow(0, bbox.x(), bbox.y(), bbox.width(), bbox.height())
                    
                    masked_pixmap = QPixmap(pixmap.size())
                    masked_pixmap.fill(Qt.transparent)
                    
                    painter = QPainter(masked_pixmap)
                    painter.setRenderHint(QPainter.Antialiasing)
                    
                    path = QPainterPath()
                    local_polygon = QPolygon([QPoint(p.x() - bbox.x(), p.y() - bbox.y()) for p in effective_points])
                    path.addPolygon(QPolygonF(local_polygon))
                    
                    painter.setClipPath(path)
                    painter.drawPixmap(0, 0, pixmap)
                    
                    # 为了明显的选中感，在外围画一圈细边框
                    pen = QPen(QColor(0, 255, 255, 150), 2)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(path)
                    painter.end()
                    
                    self._set_selected_region({
                        'pixmap': masked_pixmap,
                        'polygon': polygon,
                        'bbox': bbox,
                        'center': bbox.center(),
                        'scale': 1.0
                    })
                    self.region_state = "drawn_wait"
                    self.region_hover_frames = 0
                else:
                    self.strokes.append(stroke_obj)
            else:
                self.strokes.append(stroke_obj)

        self.is_drawing = False
        self.current_stroke = []

    def _drop_region(self):
        """将抓取的区域贴回背景或丢弃选中状态"""
        if self.selected_region:
            self.strokes.append({
                'type': 'pixmap',
                'pixmap': self.selected_region['pixmap'],
                'center': self.selected_region['center'],
                'scale': self.selected_region['scale']
            })
        self._set_selected_region(None)
        self.region_state = "none"
        self.region_hover_frames = 0
        # 冷却期：防止放下后立刻被重新拾取
        self._pickup_cooldown_until = time.time() + 1.0

    def _erase(self, pos: QPoint):
        """局部范围擦除：仅删除落入擦除圆圈内的点，将笔画从交汇处分裂为多段短线"""
        if not self.is_erasing:
            self.is_erasing = True
            self.history.append([self._copy_stroke(s) for s in self.strokes])
            if len(self.history) > self.max_history:
                self.history.pop(0)

        r = self.eraser_thickness
        r_sq = r * r
        new_strokes = []
        
        for stroke in self.strokes:
            # 处理通过抓取放置的图像图层
            if stroke.get('type') == 'pixmap':
                # 判断橡皮擦是否碰到了图像的中心区域（简单矩形碰撞）
                # 图像如果需要被擦除，可以直接整块删除。这里暂定为保留。
                # 如果你想让橡皮擦也能擦除图片，可以解开下面的距离判断。
                # center = stroke['center']
                # if (center.x() - pos.x())**2 + (center.y() - pos.y())**2 < r_sq * 4:
                #     continue # 删除这张图
                new_strokes.append(stroke)
                continue

            points = stroke.get('points', [])
            if not points:
                continue

            current_sub_points = []
            for p in points:
                dx = p.x() - pos.x()
                dy = p.y() - pos.y()
                if dx * dx + dy * dy >= r_sq:
                    # 在圆圈外，加入当前的子笔画集
                    current_sub_points.append(p)
                else:
                    # 落入擦除圆圈，切断笔画
                    if len(current_sub_points) >= 2:
                        new_strokes.append({
                            'points': current_sub_points,
                            'color': stroke['color'],
                            'thickness': stroke['thickness'],
                            'is_eraser': stroke['is_eraser']
                        })
                    current_sub_points = []
            
            # 保存该笔画最后剩余的子笔迹
            if len(current_sub_points) >= 2:
                new_strokes.append({
                    'points': current_sub_points,
                    'color': stroke['color'],
                    'thickness': stroke['thickness'],
                    'is_eraser': stroke['is_eraser']
                })

        self.strokes = new_strokes
        self.erase_pos = pos

    def _next_color(self):
        self._color_index = (self._color_index + 1) % len(self.COLORS)
        self.pen_color = self.COLORS[self._color_index]
        self.is_eraser = False
        self.floating_menu.set_selected_color(self._color_index)
        print(f"[Overlay] 颜色: {self.COLOR_NAMES[self._color_index]}")

    def _set_color(self, index: int):
        """FloatingMenu 信号：设置指定颜色"""
        if 0 <= index < len(self.COLORS):
            self._color_index = index
            self.pen_color = self.COLORS[index]
            self.is_eraser = False
            self.floating_menu.set_selected_color(index)
            print(f"[Menu] 颜色: {self.COLOR_NAMES[index]}")

    def _set_thickness(self, t: int):
        """FloatingMenu 信号：设置画笔粗细"""
        self.pen_thickness = t
        idx = MENU_THICKNESSES.index(t) if t in MENU_THICKNESSES else 1
        self.floating_menu.set_selected_thickness(idx)
        print(f"[Menu] 粗细: {t}px")

    def _on_menu_action(self, action: str):
        """FloatingMenu 信号：操作按钮"""
        if action == "clear":
            self._clear()
        elif action == "undo":
            self._undo()
        elif action == "mode":
            self._toggle_mode()
        elif action == "exit":
            print("[Overlay] 收到退出信号，程序正在关闭...")
            QApplication.quit()

    def _clear(self):
        self._stop_stroke()
        self._drop_region()
        self.history.append([self._copy_stroke(s) for s in self.strokes])
        self.strokes.clear()
        print("[Overlay] 已清空")

    def _undo(self):
        if self.history:
            self.redo_stack.append([self._copy_stroke(s) for s in self.strokes])
            self.strokes = self.history.pop()
            print(f"[Overlay] 撤销 (剩余 {len(self.history)} 步)")
        else:
            print("[Overlay] 无可撤销")

    def _redo(self):
        if self.redo_stack:
            self.history.append([self._copy_stroke(s) for s in self.strokes])
            self.strokes = self.redo_stack.pop()
            print(f"[Overlay] 重做 (剩余 {len(self.redo_stack)} 步)")
        else:
            print("[Overlay] 无可重做")

    def _toggle_mode(self):
        if self.mode == "whiteboard":
            self.mode = "annotation"
            print("[Overlay] → 屏幕批注模式")
        else:
            self.mode = "whiteboard"
            print("[Overlay] → 白板模式")

    def _trigger_llm_analysis(self):
        """V 手势：保存选区截图，调用 API 识别图片，结果在右上角白框显示"""
        if not self.selected_region:
            return

        import tempfile
        import threading
        from core.api_client import describe_image

        # 保存选区截图到临时文件
        pixmap = self.selected_region['pixmap']
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp_path = tmp.name
        tmp.close()
        pixmap.save(tmp_path, 'PNG')
        print(f"[Overlay] 选区已保存: {tmp_path}")

        # 右上角白框：显示加载中
        self._llm_result_text = "正在识别图片..."
        self._llm_result_active = True
        self.update()

        # 后台线程调用 API，避免阻塞 UI
        def _analyze():
            try:
                result = describe_image(tmp_path)
                self._llm_result_text = result
            except Exception as e:
                self._llm_result_text = f"识别失败: {str(e)[:100]}"
            self.update()

        threading.Thread(target=_analyze, daemon=True).start()

    def _update_camera(self):
        """更新摄像头预览"""
        if self.engine is None:
            return
        frame = self.engine.get_frame()  # 已经是由后台线程转换好的 RGB 数据
        if frame is None:
            return

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        # 创建并深度复制 QImage，避免在 GUI 线程中调用 cv2.cvtColor 和创建 QPixmap
        self.camera_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.update()

    # ===================== 键盘 =====================

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Q:
            print("[Overlay] 退出")
            QApplication.quit()
        elif key == Qt.Key_E:
            self.is_eraser = not self.is_eraser
            print(f"[Overlay] {'橡皮擦' if self.is_eraser else '画笔'}")
        elif key == Qt.Key_N:
            self._clear()
        elif key == Qt.Key_C:
            self._clear()
        elif key == Qt.Key_Z:
            self._undo()
        elif key == Qt.Key_A:
            self._toggle_mode()
        elif key == Qt.Key_D:
            pass

        self.update()

    # ===================== 绘制 =====================

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # ===== 半透明背景（白板模式时）=====
        if self.mode == "whiteboard":
            painter.fillRect(self.rect(), QColor(255, 255, 255, 230))
        else:
            # 批注模式下使用极其微弱的透明填充 (Alpha = 1)，强制 Windows DWM 启用 GPU 硬件加速，解决卡顿
            painter.fillRect(self.rect(), QColor(0, 0, 0, 1))

        # ===== 绘制所有已完成的笔画 =====
        for stroke in self.strokes:
            self._draw_stroke(painter, stroke)

        # ===== 绘制当前笔画 =====
        if self.current_stroke:
            current_stroke_obj = {
                'points': self.current_stroke,
                'color': self.pen_color,
                'thickness': self.pen_thickness,
                'is_eraser': self.is_eraser
            }
            self._draw_stroke(painter, current_stroke_obj)

        # ===== 激光笔拖影特效 =====
        if len(self.laser_trail) > 1:
            for i in range(len(self.laser_trail) - 1):
                p1 = self.laser_trail[i]
                p2 = self.laser_trail[i+1]
                # 越老的点透明度越低
                alpha = int(255 * (i / len(self.laser_trail)))
                # 画笔变粗一点，带有发光感
                pen = QPen(QColor(0, 255, 255, alpha), 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(p1, p2)

        # ===== 擦除圆圈 =====
        if self.erase_pos and self.five_finger_eraser:
            painter.setPen(QPen(QColor(200, 200, 200), 2))
            painter.drawEllipse(self.erase_pos, self.eraser_thickness, self.eraser_thickness)

        # ===== 摄像头预览（左上角 200x150）=====
        if hasattr(self, 'camera_image') and self.camera_image:
            thumb_w, thumb_h = 200, 150
            x0 = 10
            y0 = 40
            painter.fillRect(QRect(x0 - 2, y0 - 2, thumb_w + 4, thumb_h + 4),
                           QColor(0, 0, 0, 100))
            painter.drawImage(QRect(x0, y0, thumb_w, thumb_h), self.camera_image)
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x0, y0, thumb_w, thumb_h)

        # ===== 底部状态栏 =====
        self._draw_status_bar(painter)

        # ===== 绘制状态提示
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        painter.drawText(10, 30, f"Mode: {self.mode}")
        painter.drawText(10, 60, f"Status: {self.status_text}")
        
        # 绘制动态手势超级大字反馈
        if self.dynamic_feedback_timer > 0:
            self.dynamic_feedback_timer -= 1
            painter.setPen(QColor(0, 255, 255, min(255, self.dynamic_feedback_timer * 10)))
            painter.setFont(QFont("Arial", 60, QFont.Bold))
            rect = QRect(0, 0, self.screen_w, self.screen_h)
            painter.drawText(rect, Qt.AlignCenter, self.dynamic_feedback_text)

        # ===== 右上角 LLM 分析结果白框（自适应高度）=====
        if self._llm_result_active and self._llm_result_text:
            from PyQt5.QtGui import QFontMetrics
            box_w = 450
            box_x = self.screen_w - box_w - 20
            box_y = 20
            padding = 20
            header_h = 30

            # 计算文本所需高度
            painter.setFont(QFont("Arial", 10))
            fm = QFontMetrics(painter.font())
            text_area_w = box_w - padding * 2
            text_rect = QRect(0, 0, text_area_w, 10000)
            text_h = fm.boundingRect(text_rect, Qt.TextWordWrap, self._llm_result_text).height()
            box_h = header_h + text_h + padding * 2
            box_h = max(box_h, 80)  # 最小高度

            painter.fillRect(QRect(box_x, box_y, box_w, box_h), QColor(255, 255, 255, 220))
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRect(box_x, box_y, box_w, box_h))
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QFont("Arial", 12, QFont.Bold))
            painter.drawText(box_x + 10, box_y + 22, "AI Analysis")
            painter.setFont(QFont("Arial", 10))
            text_draw_rect = QRect(box_x + padding, box_y + header_h, text_area_w, text_h + 10)
            painter.drawText(text_draw_rect, Qt.TextWordWrap, self._llm_result_text)

        # ===== 被抓取的区域 =====
        if self.selected_region:
            pixmap = self.selected_region['pixmap']
            center = self.selected_region['center']
            scale = self.selected_region['scale']
            
            painter.save()
            painter.translate(center)
            painter.scale(scale, scale)
            painter.drawPixmap(-pixmap.width() // 2, -pixmap.height() // 2, pixmap)
            painter.restore()

            # Hover progress ring
            if self.region_hover_frames > 0 and self.laser_pos:
                pen = QPen(QColor(0, 255, 255), 4)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                r = 20
                rect = QRect(self.laser_pos.x() - r, self.laser_pos.y() - r, r * 2, r * 2)
                max_frames = self.HOVER_SELECT_FRAMES if self.region_state == "drawn_wait" else 45
                span_angle = int((self.region_hover_frames / max_frames) * 360 * 16)
                painter.drawArc(rect, 90 * 16, span_angle)

        # ===== 拾取已放置选区的进度环 =====
        if self._pickup_hover_frames > 0 and self.laser_pos and self.selected_region is None:
            pen = QPen(QColor(255, 200, 0), 4)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            r = 25
            rect = QRect(self.laser_pos.x() - r, self.laser_pos.y() - r, r * 2, r * 2)
            span_angle = int((self._pickup_hover_frames / 30) * 360 * 16)
            painter.drawArc(rect, 90 * 16, span_angle)

        # ===== 悬浮菜单 =====
        laser_x = self.laser_pos.x() if self.laser_pos else None
        self.floating_menu.tick(laser_x)
        self.floating_menu.paint(painter)

        # ===== 激光笔（移至菜单上方绘制，解决遮挡问题）=====
        if self.laser_pos and not self.is_drawing and not self.five_finger_eraser:
            # 绘制代表画笔颜色和粗细的实心圆圈光标
            radius = max(4, self.pen_thickness // 2)
            painter.setPen(QPen(QColor(128, 128, 128, 180), 1))
            painter.setBrush(QBrush(self.pen_color))
            painter.drawEllipse(self.laser_pos, radius, radius)

            # 捏合确认进度环（如果正在开始画画捏合）
            if 0.0 < self._pinch_progress < 1.0:
                pen = QPen(QColor(255, 200, 0), 3)
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                r = radius + 8
                rect = QRect(self.laser_pos.x() - r, self.laser_pos.y() - r, r * 2, r * 2)
                span_angle = int(self._pinch_progress * 360 * 16)
                painter.drawArc(rect, 90 * 16, span_angle)

        painter.end()

    def _draw_stroke(self, painter: QPainter, stroke: dict):
        """绘制一条笔画或图像图层"""
        if stroke.get('type') == 'pixmap':
            pixmap = stroke['pixmap']
            center = stroke['center']
            scale = stroke['scale']
            painter.save()
            painter.translate(center)
            painter.scale(scale, scale)
            painter.drawPixmap(-pixmap.width() // 2, -pixmap.height() // 2, pixmap)
            painter.restore()
            return

        points = stroke.get('points', [])
        if len(points) < 2:
            return

        if stroke['is_eraser']:
            pen = QPen(QColor(255, 255, 255, 0), stroke['thickness'])
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
        else:
            pen = QPen(stroke['color'], stroke['thickness'])
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)

        for i in range(1, len(points)):
            painter.drawLine(points[i - 1], points[i])

    def _draw_status_bar(self, painter: QPainter):
        """底部状态栏"""
        bar_h = 40
        y0 = self.screen_h - bar_h

        painter.fillRect(QRect(0, y0, self.screen_w, bar_h), QColor(30, 30, 30, 180))

        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Microsoft YaHei", 11))

        color_name = self.COLOR_NAMES[self._color_index]
        mode = "橡皮" if self.is_eraser else "画笔"
        drawing = "擦除中" if self.five_finger_eraser else ("画画中" if self.is_drawing else "待机")
        bg = "白板" if self.mode == "whiteboard" else "批注"

        text = f"{bg} | {mode} | {color_name} | 粗细:{self.pen_thickness} | {drawing} | {self.fps_text} | E:橡皮 N:清空 Z:撤销 Q:退出"
        painter.drawText(10, y0 + 26, text)


# 需要 import cv2（用于摄像头帧转换）
import cv2


def create_overlay_window(gesture_queue: queue.Queue, engine=None) -> OverlayWindow:
    """创建覆盖窗口"""
    window = OverlayWindow(gesture_queue, engine)
    window.showFullScreen()
    return window
