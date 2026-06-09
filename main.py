"""
教室白板系统 — 前后端分离架构
后端：MediaPipe 手势引擎（独立线程）
前端：PyQt5 透明全屏窗口（画线 + 摄像头预览）
"""

import sys
import queue
import signal

# Ctrl+C 优雅退出
signal.signal(signal.SIGINT, signal.SIG_DFL)


def main():
    from PyQt5.QtWidgets import QApplication

    from core.gesture_engine import GestureEngine
    from core.overlay_window import OverlayWindow

    print("=" * 50)
    print("  教室白板系统 - 透明全屏批注")
    print("=" * 50)
    print("  手势：")
    print("    食指+拇指捏合 -> 画画/批注")
    print("    五指张开 -> 橡皮擦")
    print("    双手拍手 -> 切换颜色")
    print("  键盘：")
    print("    E:橡皮  N/C:清空  Z:撤销  A:切换模式  Q:退出")
    print("=" * 50)

    # 创建 PyQt 应用
    app = QApplication(sys.argv)

    # 创建手势队列
    gesture_queue = queue.Queue(maxsize=2)

    # 启动手势引擎（后台线程）
    engine = GestureEngine(camera_id=0, output_queue=gesture_queue)
    engine.start()

    # 创建透明覆盖窗口
    window = OverlayWindow(gesture_queue, engine)
    window.showFullScreen()

    # 运行 Qt 事件循环
    try:
        sys.exit(app.exec_())
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
