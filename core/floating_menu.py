"""
悬浮菜单 — 专为隔空手势交互设计
嵌入 OverlayWindow 内部绘制，全部竖排单列布局
"""

import time
from typing import Optional, List
from PyQt5.QtCore import Qt, QRect, QRectF, QPoint
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QFont, QPainterPath,
    QRadialGradient, QLinearGradient
)


# ==============================================================================
# 配置常量
# ==============================================================================

PANEL_WIDTH = 100           # 面板宽度 (从80调大到100)
PANEL_MARGIN = 16           # 面板内边距
PANEL_RADIUS = 24           # 面板圆角半径
PANEL_GAP = 14              # 按钮间距 (防误触拉大间距)

COLOR_BTN_SIZE = 64         # 色块按钮尺寸（圆形，从50调大到64）
THICKNESS_BTN_H = 46        # 粗细按钮高度 (从36调大到46)
ACTION_BTN_H = 50           # 操作按钮高度 (从40调大到50)

HOVER_SCALE = 1.15          # 悬停放大比例
CLICK_CONFIRM_MS = 300      # 点击闪光持续时间

# 颜色定义
COLORS = [
    (QColor(255, 59, 48),   "红"),
    (QColor(0, 122, 255),   "蓝"),
    (QColor(52, 199, 89),   "绿"),
    (QColor(255, 149, 0),   "橙"),
    (QColor(175, 82, 222),  "紫"),
    (QColor(28, 28, 30),    "黑"),
]

THICKNESSES = [2, 5, 12]
THICKNESS_LABELS = ["细", "中", "粗"]


# ==============================================================================
# 按钮数据模型
# ==============================================================================

class ButtonItem:

    def __init__(self, btn_id: str, rect: QRect, label: str = "",
                 color: Optional[QColor] = None, btn_type: str = "action"):
        self.btn_id = btn_id
        self.rect = rect
        self.label = label
        self.color = color
        self.btn_type = btn_type

        self.hover = False
        self.hover_progress = 0.0
        self.flash = False
        self.flash_start = 0.0
        self.hover_start_time = 0.0
        self.triggered_this_hover = False

    def contains(self, x: int, y: int) -> bool:
        return self.rect.contains(x, y)


# ==============================================================================
# 悬浮菜单
# ==============================================================================

class FloatingMenu:

    def __init__(self, screen_w: int = 1920, screen_h: int = 1080):
        self.screen_w = screen_w
        self.screen_h = screen_h

        # 按钮列表
        self.buttons: List[ButtonItem] = []
        self._build_buttons()

        # 面板高度（根据按钮实际位置动态算）
        self.panel_height = self._calc_panel_height()

        # 伸缩动画坐标
        self.expanded_x = screen_w - PANEL_WIDTH - 16
        self.collapsed_x = screen_w - 15  # 仅留出 15px 边缘在屏幕内用于手势接触展开
        self.current_x = self.collapsed_x
        self.panel_x = self.current_x

        # 整体垂直居中对齐
        self.panel_y = (screen_h - self.panel_height) // 2

        self.is_expanded = False

        # 当前选中状态
        self._selected_color = 0
        self._selected_thickness = 1

        # 防重复触发
        self._last_trigger_time = 0.0
        self._trigger_cooldown = 0.5

        # 当前悬停按钮
        self._hover_btn: Optional[ButtonItem] = None

    # ── 面板高度 ──

    def _calc_panel_height(self) -> int:
        if not self.buttons:
            return 200
        last_btn = self.buttons[-1]
        return last_btn.rect.bottom() + PANEL_MARGIN

    # ── 构建按钮（全部竖排）──

    def _build_buttons(self):
        self.buttons.clear()
        cx = PANEL_WIDTH // 2  # 水平居中线
        y = PANEL_MARGIN

        # ── 颜色 ──
        y += 4  # 微调
        for i, (color, name) in enumerate(COLORS):
            r = COLOR_BTN_SIZE // 2
            rect = QRect(cx - r, y, COLOR_BTN_SIZE, COLOR_BTN_SIZE)
            self.buttons.append(ButtonItem(
                btn_id=f"color_{i}", rect=rect,
                label=name, color=color, btn_type="color"
            ))
            y += COLOR_BTN_SIZE + PANEL_GAP

        # ── 分隔 ──
        y += 4

        # ── 粗细 ──
        for i, (t, label) in enumerate(zip(THICKNESSES, THICKNESS_LABELS)):
            bw = PANEL_WIDTH - PANEL_MARGIN * 2
            rect = QRect(PANEL_MARGIN, y, bw, THICKNESS_BTN_H)
            self.buttons.append(ButtonItem(
                btn_id=f"thick_{t}", rect=rect,
                label=label, btn_type="thickness"
            ))
            y += THICKNESS_BTN_H + PANEL_GAP

        # ── 分隔 ──
        y += 4

        # ── 操作按钮 ──
        bw = PANEL_WIDTH - PANEL_MARGIN * 2
        rect_clear = QRect(PANEL_MARGIN, y, bw, ACTION_BTN_H)
        self.buttons.append(ButtonItem(
            btn_id="clear", rect=rect_clear,
            label="清空", btn_type="action"
        ))
        y += ACTION_BTN_H + PANEL_GAP

        rect_undo = QRect(PANEL_MARGIN, y, bw, ACTION_BTN_H)
        self.buttons.append(ButtonItem(
            btn_id="undo", rect=rect_undo,
            label="撤销", btn_type="action"
        ))
        y += ACTION_BTN_H + PANEL_GAP

        rect_mode = QRect(PANEL_MARGIN, y, bw, ACTION_BTN_H)
        self.buttons.append(ButtonItem(
            btn_id="mode", rect=rect_mode,
            label="模式", btn_type="action"
        ))
        y += ACTION_BTN_H + PANEL_GAP

        rect_exit = QRect(PANEL_MARGIN, y, bw, ACTION_BTN_H)
        self.buttons.append(ButtonItem(
            btn_id="exit", rect=rect_exit,
            label="退出", btn_type="action"
        ))

    # ── 公开接口 ──

    def set_selected_color(self, index: int):
        if 0 <= index < len(COLORS):
            self._selected_color = index

    def set_selected_thickness(self, index: int):
        if 0 <= index < len(THICKNESSES):
            self._selected_thickness = index

    def in_menu_area(self, screen_x: int) -> bool:
        if not self.is_expanded:
            # 缩回状态下，仅屏幕最右侧 40px 的区域能触发展开
            return screen_x >= self.screen_w - 40
        else:
            # 展开状态下，在菜单左侧 50px 到菜单右侧都属于热区
            return screen_x >= self.panel_x - 50

    # ── 核心：hit_test ──

    def hit_test(self, screen_x: int, screen_y: int,
                 on_color=None, on_thickness=None, on_action=None) -> Optional[str]:
        lx = screen_x - self.panel_x
        ly = screen_y - self.panel_y

        in_panel = (-30 <= lx <= PANEL_WIDTH + 30 and
                    -30 <= ly <= self.panel_height + 30)

        # 更新悬停
        hovered_btn = None
        now = time.time()

        for btn in self.buttons:
            was_hover = btn.hover
            btn.hover = in_panel and btn.contains(lx, ly)
            if btn.hover:
                hovered_btn = btn
                if not was_hover:
                    btn.hover_start_time = now
                    btn.triggered_this_hover = False
            else:
                btn.hover_start_time = 0.0
                btn.triggered_this_hover = False

        self._hover_btn = hovered_btn

        # 处理点击：悬停 1.5 秒触发
        triggered = None

        if hovered_btn and not hovered_btn.triggered_this_hover:
            elapsed = now - hovered_btn.hover_start_time
            required_time = 3.0 if hovered_btn.btn_id == "exit" else 1.5
            if elapsed >= required_time:
                triggered = hovered_btn.btn_id
                hovered_btn.triggered_this_hover = True
                hovered_btn.flash = True
                hovered_btn.flash_start = now
                self._last_trigger_time = now

                if hovered_btn.btn_type == "color":
                    idx = int(hovered_btn.btn_id.split("_")[1])
                    self._selected_color = idx
                    if on_color:
                        on_color(idx)
                elif hovered_btn.btn_type == "thickness":
                    t = int(hovered_btn.btn_id.split("_")[1])
                    idx = THICKNESSES.index(t) if t in THICKNESSES else 1
                    self._selected_thickness = idx
                    if on_thickness:
                        on_thickness(t)
                elif hovered_btn.btn_type == "action":
                    if on_action:
                        on_action(hovered_btn.btn_id)

        # 光标离开面板
        if not in_panel:
            for btn in self.buttons:
                btn.hover = False
                btn.hover_start_time = 0.0
                btn.triggered_this_hover = False

        return triggered

    def get_hover_progress(self) -> float:
        """返回当前悬停按钮的进度 (0.0 ~ 1.0)"""
        if self._hover_btn and self._hover_btn.hover and not self._hover_btn.triggered_this_hover:
            now = time.time()
            elapsed = now - self._hover_btn.hover_start_time
            required_time = 3.0 if self._hover_btn.btn_id == "exit" else 1.5
            return min(elapsed / required_time, 1.0)
        return 0.0

    # ── 动画 ──

    def tick(self, laser_x: Optional[int] = None):
        # 1. 更新伸缩逻辑
        if laser_x is not None:
            if laser_x >= self.screen_w - 40 or self.in_menu_area(laser_x):
                self.is_expanded = True
            elif laser_x < self.screen_w - PANEL_WIDTH - 120:
                self.is_expanded = False
        else:
            # 手势丢失，自动收回
            self.is_expanded = False

        # 平滑插值动画更新面板 X 轴位置
        target_pos_x = self.expanded_x if self.is_expanded else self.collapsed_x
        self.current_x += (target_pos_x - self.current_x) * 0.18
        self.panel_x = int(self.current_x)

        # 2. 更新按钮悬停缩放与闪烁动画
        for btn in self.buttons:
            target = 1.0 if btn.hover else 0.0
            btn.hover_progress += (target - btn.hover_progress) * 0.25

            if btn.flash:
                elapsed = time.time() - btn.flash_start
                if elapsed > CLICK_CONFIRM_MS / 1000:
                    btn.flash = False

    # ── 绘制 ──

    def paint(self, painter: QPainter):
        painter.save()
        painter.translate(self.panel_x, self.panel_y)

        # 面板背景
        panel_rect = QRectF(0, 0, PANEL_WIDTH, self.panel_height)
        self._draw_panel_bg(painter, panel_rect)

        # 按钮
        for btn in self.buttons:
            self._draw_button(painter, btn)

        # 悬停发光
        if self._hover_btn and self._hover_btn.hover_progress > 0.05:
            self._draw_glow(painter, self._hover_btn)

        painter.restore()

    def _draw_panel_bg(self, painter: QPainter, rect: QRectF):
        path = QPainterPath()
        path.addRoundedRect(rect, PANEL_RADIUS, PANEL_RADIUS)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(25, 25, 30, 190))
        painter.drawPath(path)

        border = QPainterPath()
        border.addRoundedRect(rect.adjusted(1, 1, -1, -1), PANEL_RADIUS - 1, PANEL_RADIUS - 1)
        painter.setPen(QPen(QColor(255, 255, 255, 25), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(border)

        hl = QLinearGradient(0, 0, 0, 60)
        hl.setColorAt(0, QColor(255, 255, 255, 18))
        hl.setColorAt(1, QColor(255, 255, 255, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(hl)
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), PANEL_RADIUS, PANEL_RADIUS)

    def _draw_button(self, painter: QPainter, btn: ButtonItem):
        scale = 1.0 + (HOVER_SCALE - 1.0) * btn.hover_progress
        center = btn.rect.center()

        painter.save()
        painter.translate(center)
        painter.scale(scale, scale)
        painter.translate(-center)

        if btn.btn_type == "color":
            self._draw_color_btn(painter, btn)
        elif btn.btn_type == "thickness":
            self._draw_thickness_btn(painter, btn)
        elif btn.btn_type == "action":
            self._draw_action_btn(painter, btn)

        # 绘制悬停进度环/进度条
        now = time.time()
        if btn.hover and not btn.triggered_this_hover:
            required_time = 3.0 if btn.btn_id == "exit" else 1.5
            prog = min((now - btn.hover_start_time) / required_time, 1.0)
            if prog > 0.0:
                if btn.btn_type == "color":
                    r = btn.rect.width() // 2
                    pen = QPen(QColor(255, 200, 0, 220), 3)
                    pen.setCapStyle(Qt.RoundCap)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    rect = QRectF(center.x() - r - 6, center.y() - r - 6, (r + 6) * 2, (r + 6) * 2)
                    span_angle = int(prog * 360 * 16)
                    painter.drawArc(rect, 90 * 16, -span_angle)  # 顺时针
                else:
                    rect = btn.rect
                    bar_h = 4
                    bar_w = int(rect.width() * prog)
                    bar_rect = QRect(rect.left(), rect.bottom() - bar_h, bar_w, bar_h)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(255, 200, 0, 220))
                    painter.drawRoundedRect(bar_rect, 2, 2)

        painter.restore()

    def _draw_color_btn(self, painter: QPainter, btn: ButtonItem):
        r = btn.rect.width() // 2
        center = btn.rect.center()
        idx = int(btn.btn_id.split("_")[1])
        is_selected = (idx == self._selected_color)

        if is_selected:
            painter.setPen(QPen(QColor(255, 255, 255, 220), 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(center, r + 4, r + 4)

        painter.setPen(Qt.NoPen)
        painter.setBrush(btn.color)
        painter.drawEllipse(center, r, r)

        hl = QRadialGradient(center.x() - r * 0.3, center.y() - r * 0.3, r * 0.8)
        hl.setColorAt(0, QColor(255, 255, 255, 60))
        hl.setColorAt(1, QColor(255, 255, 255, 0))
        painter.setBrush(hl)
        painter.drawEllipse(center, r, r)

        if btn.flash:
            elapsed = time.time() - btn.flash_start
            alpha = max(0, int(180 * (1 - elapsed / (CLICK_CONFIRM_MS / 1000))))
            painter.setBrush(QColor(255, 255, 255, alpha))
            painter.drawEllipse(center, r, r)

    def _draw_thickness_btn(self, painter: QPainter, btn: ButtonItem):
        rect = btn.rect
        t = int(btn.btn_id.split("_")[1])
        idx = THICKNESSES.index(t) if t in THICKNESSES else 1
        is_selected = (idx == self._selected_thickness)

        bg = QColor(60, 60, 65, 200) if not is_selected else QColor(80, 130, 255, 200)
        painter.setPen(Qt.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(rect, 10, 10)

        if is_selected:
            painter.setPen(QPen(QColor(255, 255, 255, 180), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 9, 9)

        # 粗细示意横线
        line_y = rect.center().y()
        line_x0 = rect.left() + 10
        line_x1 = rect.right() - 10
        painter.setPen(QPen(QColor(255, 255, 255, 230), t))
        painter.drawLine(QPoint(line_x0, line_y), QPoint(line_x1, line_y))

        # 文字
        painter.setPen(QColor(200, 200, 205, 200))
        painter.setFont(QFont("Microsoft YaHei", 7))
        painter.drawText(rect, Qt.AlignBottom | Qt.AlignHCenter, btn.label)

    def _draw_action_btn(self, painter: QPainter, btn: ButtonItem):
        rect = btn.rect
        if btn.btn_id == "exit":
            bg = QColor(139, 0, 0, 200)  # 深红
        elif btn.btn_id == "clear":
            bg = QColor(200, 50, 50, 180)  # 红色
        else:
            bg = QColor(60, 60, 65, 200)   # 深灰

        painter.setPen(Qt.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(rect, 10, 10)

        if btn.btn_id == "exit":
            border = QColor(255, 100, 100, 120)
        elif btn.btn_id == "clear":
            border = QColor(255, 100, 100, 80)
        else:
            border = QColor(255, 255, 255, 30)
        painter.setPen(QPen(border, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 9, 9)

        painter.setPen(QColor(255, 255, 255, 230))
        painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, btn.label)

        if btn.flash:
            elapsed = time.time() - btn.flash_start
            alpha = max(0, int(150 * (1 - elapsed / (CLICK_CONFIRM_MS / 1000))))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, alpha))
            painter.drawRoundedRect(rect, 10, 10)

    def _draw_glow(self, painter: QPainter, btn: ButtonItem):
        center = btn.rect.center()
        scale = 1.0 + (HOVER_SCALE - 1.0) * btn.hover_progress
        r = int(btn.rect.width() * scale * 0.7)

        glow = QRadialGradient(center.x(), center.y(), r)
        glow.setColorAt(0, QColor(100, 180, 255, int(50 * btn.hover_progress)))
        glow.setColorAt(0.6, QColor(100, 180, 255, int(20 * btn.hover_progress)))
        glow.setColorAt(1, QColor(100, 180, 255, 0))

        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(center, r, r)

    def is_animating(self) -> bool:
        """检查菜单面板或按钮是否正处于动画中"""
        target_pos_x = self.expanded_x if self.is_expanded else self.collapsed_x
        if abs(self.current_x - target_pos_x) > 0.5:
            return True
        for btn in self.buttons:
            target_prog = 1.0 if btn.hover else 0.0
            if abs(btn.hover_progress - target_prog) > 0.01:
                return True
            if btn.flash:
                return True
        return False
