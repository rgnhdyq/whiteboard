"""
智能教学白板 — 启动器 V2 (高级仪表盘版)
功能：配置 API、启动白板、查看控制台日志
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_CUDA_ARCH_LIST'] = ''

import sys

try:
    import torch
except ImportError:
    pass

import json
import time
import threading
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QFormLayout, QComboBox, 
    QGraphicsDropShadowEffect, QFrame, QMessageBox, QSpacerItem, QSizePolicy,
    QSizeGrip, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QCursor

if getattr(sys, 'frozen', False):
    CONFIG_PATH = os.path.join(os.path.dirname(sys.executable), 'api_config.json')
else:
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_config.json')

# 预设服务商
PROVIDERS = {
    'minimax': {
        'name': 'MiniMax',
        'base_url': 'https://api.minimaxi.com/v1',
        'models': ['MiniMax-M3', 'MiniMax-M2.7', 'MiniMax-M2.7-highspeed']
    },
    'openai': {
        'name': 'OpenAI',
        'base_url': 'https://api.openai.com/v1',
        'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-vision-preview']
    },
    'deepseek': {
        'name': 'DeepSeek',
        'base_url': 'https://api.deepseek.com/v1',
        'models': ['deepseek-chat']
    },
    'moonshot': {
        'name': 'Moonshot (Kimi)',
        'base_url': 'https://api.moonshot.cn/v1',
        'models': ['moonshot-v1-8k', 'moonshot-v1-32k', 'moonshot-v1-128k']
    },
    'custom': {
        'name': '自定义',
        'base_url': '',
        'models': []
    }
}

class LogRedirector(QObject):
    """将子进程输出重定向到 GUI"""
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.log_signal.connect(self._append_log)
        self._log_widget = None

    def set_widget(self, widget):
        self._log_widget = widget

    def _append_log(self, text):
        if self._log_widget:
            self._log_widget.append(text)
            scrollbar = self._log_widget.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())


class TitleBar(QWidget):
    """自定义无边框标题栏"""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(16, 8, 16, 8)
        self.layout.setSpacing(10)
        
        self.title_label = QLabel("✨ AI Whiteboard Console V2")
        self.title_label.setStyleSheet("color: #339af0; font-size: 14px; font-weight: bold; font-family: 'Segoe UI';")
        
        self.layout.addWidget(self.title_label)
        self.layout.addStretch()
        
        # 最小化按钮
        self.min_btn = QPushButton("─")
        self.min_btn.setFixedSize(30, 30)
        self.min_btn.setStyleSheet("""
            QPushButton { background-color: transparent; color: #868e96; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #e9ecef; color: #212529; }
        """)
        self.min_btn.clicked.connect(self.parent.showMinimized)
        
        # 关闭按钮
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.setStyleSheet("""
            QPushButton { background-color: transparent; color: #868e96; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #ff8787; color: #ffffff; }
        """)
        self.close_btn.clicked.connect(self.parent.close)
        
        self.layout.addWidget(self.min_btn)
        self.layout.addWidget(self.close_btn)
        
        self.start_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.start_pos is not None:
            delta = event.globalPos() - self.start_pos
            self.parent.move(self.parent.x() + delta.x(), self.parent.y() + delta.y())
            self.start_pos = event.globalPos()

    def mouseReleaseEvent(self, event):
        self.start_pos = None


class LauncherWindowV2(QWidget):
    """新版高级启动器窗口"""
    def __init__(self):
        super().__init__()
        self.process = None
        self.log_redirector = LogRedirector()
        
        # 窗口基本设置
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(1200, 750)
        
        self._init_ui()
        self._load_config()

    def _init_ui(self):
        # 顶级布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 主容器（带圆角和阴影底色）
        self.container = QFrame()
        self.container.setObjectName("MainContainer")
        self.container.setStyleSheet("""
            #MainContainer {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 12px;
            }
            QLabel {
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            }
        """)
        
        # 添加窗口阴影
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 4)
        self.container.setGraphicsEffect(shadow)
        
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        
        # ==================== 顶部标题栏 ====================
        self.title_bar = TitleBar(self)
        container_layout.addWidget(self.title_bar)
        
        # ==================== 核心内容区 (左右分栏) ====================
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(20, 10, 20, 20)
        content_layout.setSpacing(20)
        
        # --- 左侧：控制与设置面板 ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(20)
        
        # 欢迎卡片
        welcome_card = QFrame()
        welcome_card.setStyleSheet("background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px;")
        welcome_layout = QVBoxLayout(welcome_card)
        welcome_title = QLabel('🎓 智能教学白板')
        welcome_title.setStyleSheet("color: #4dabf7; font-size: 26px; font-weight: 800; letter-spacing: 1px;")
        welcome_sub = QLabel('手势控制 · 屏幕批注 · AI 辅助分析')
        welcome_sub.setStyleSheet("color: #868e96; font-size: 13px;")
        welcome_layout.addWidget(welcome_title)
        welcome_layout.addWidget(welcome_sub)
        left_panel.addWidget(welcome_card)
        
        # API设置卡片
        api_card = QFrame()
        arrow_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'combo_arrow.png').replace('\\', '/')
        api_card.setStyleSheet(f"""
            QFrame {{
                background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 8px;
            }}
            QLineEdit, QComboBox {{
                background-color: #f8f9fa;
                border: 1px solid #ced4da;
                border-radius: 6px;
                padding: 8px 12px;
                color: #212529;
                font-size: 16px;
            }}
            QLineEdit:focus, QComboBox:focus {{
                border: 1px solid #339af0;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 30px;
                border: none;
            }}
            QComboBox::down-arrow {{
                image: url("{arrow_path}");
                width: 12px;
                height: 12px;
            }}
            QLabel {{
                color: #495057;
                font-weight: bold;
                font-size: 15px;
                border: none;
            }}
        """)
        api_layout = QFormLayout(api_card)
        api_layout.setContentsMargins(20, 20, 20, 20)
        api_layout.setSpacing(16)
        
        # 控件初始化
        self.provider_combo = QComboBox()
        for key, val in PROVIDERS.items():
            self.provider_combo.addItem(val['name'], key)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText('输入 API Key...')
        self.api_key_input.setEchoMode(QLineEdit.Password)
        
        self.show_key_btn = QPushButton('👁')
        self.show_key_btn.setFixedSize(36, 36)
        self.show_key_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.show_key_btn.setStyleSheet("""
            QPushButton { background-color: #e9ecef; color: #495057; border-radius: 6px; font-size: 16px; border: none; }
            QPushButton:hover { background-color: #dee2e6; }
        """)
        self.show_key_btn.clicked.connect(self._toggle_key_visibility)
        
        key_h_layout = QHBoxLayout()
        key_h_layout.setSpacing(8)
        key_h_layout.addWidget(self.api_key_input)
        key_h_layout.addWidget(self.show_key_btn)
        
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        
        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText('https://api.minimaxi.com/v1')
        
        api_layout.addRow('服务商', self.provider_combo)
        api_layout.addRow('API Key', key_h_layout)
        api_layout.addRow('模型名称', self.model_combo)
        api_layout.addRow('Base URL', self.base_url_input)
        
        left_panel.addWidget(api_card)
        
        # 按钮控制区卡片
        btn_card = QFrame()
        btn_card.setStyleSheet("""
            QFrame { background-color: transparent; }
            QPushButton {
                border-radius: 8px;
                padding: 12px 0px;
                font-size: 16px;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QPushButton#runBtn {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4dabf7, stop:1 #339af0);
                color: #ffffff;
            }
            QPushButton#runBtn:hover { background-color: #74c0fc; }
            QPushButton#stopBtn {
                background-color: #ff8787;
                color: #ffffff;
            }
            QPushButton#stopBtn:hover { background-color: #ffa8a8; }
            QPushButton#stopBtn:disabled { background-color: #dee2e6; color: #adb5bd; }
            QPushButton#saveBtn {
                background-color: transparent;
                border: 2px solid #339af0;
                color: #339af0;
            }
            QPushButton#saveBtn:hover { background-color: #339af0; color: #ffffff; }
        """)
        btn_layout = QHBoxLayout(btn_card)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(12)
        
        self.save_btn = QPushButton('保存配置')
        self.save_btn.setObjectName('saveBtn')
        self.save_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.save_btn.clicked.connect(self._save_config)
        
        self.run_btn = QPushButton('🚀 启动引擎')
        self.run_btn.setObjectName('runBtn')
        self.run_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.run_btn.clicked.connect(self._run_whiteboard)
        
        self.stop_btn = QPushButton('⏹ 停止运行')
        self.stop_btn.setObjectName('stopBtn')
        self.stop_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_whiteboard)
        
        btn_layout.addWidget(self.save_btn, 1)
        btn_layout.addWidget(self.run_btn, 2)
        btn_layout.addWidget(self.stop_btn, 1)
        
        left_panel.addWidget(btn_card)
        left_panel.addStretch()
        
        # 左侧占屏幕的 35%
        content_layout.addLayout(left_panel, 35)
        
        # --- 右侧：终端日志面板 ---
        right_panel = QFrame()
        right_panel.setStyleSheet("""
            QFrame {
                background-color: #f1f3f5;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(2, 2, 2, 2)
        
        # 伪终端标题栏
        terminal_header = QFrame()
        terminal_header.setStyleSheet("background-color: #e9ecef; border-radius: 8px; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px;")
        terminal_header.setFixedHeight(36)
        th_layout = QHBoxLayout(terminal_header)
        th_layout.setContentsMargins(12, 0, 12, 0)
        
        dot_layout = QHBoxLayout()
        dot_layout.setSpacing(6)
        for color in ['#f38ba8', '#f9e2af', '#a6e3a1']:
            dot = QLabel()
            dot.setFixedSize(12, 12)
            dot.setStyleSheet(f"background-color: {color}; border-radius: 6px;")
            dot_layout.addWidget(dot)
        th_layout.addLayout(dot_layout)
        
        th_title = QLabel("SYSTEM CONSOLE")
        th_title.setStyleSheet("color: #868e96; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        th_title.setAlignment(Qt.AlignCenter)
        th_layout.addStretch()
        th_layout.addWidget(th_title)
        th_layout.addStretch()
        
        self.save_log_btn = QPushButton("保存日志")
        self.save_log_btn.setStyleSheet("color: #339af0; background: transparent; font-size: 13px; font-weight: bold;")
        self.save_log_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.save_log_btn.clicked.connect(self._save_log_to_file)
        th_layout.addWidget(self.save_log_btn)

        self.clear_btn = QPushButton("清空")
        self.clear_btn.setStyleSheet("color: #ff8787; background: transparent; font-size: 13px; font-weight: bold;")
        self.clear_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.clear_btn.clicked.connect(lambda: self.log_text.clear())
        th_layout.addWidget(self.clear_btn)
        
        right_layout.addWidget(terminal_header)
        
        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                border: none;
                color: #495057;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 14px;
                padding: 10px;
            }
            QScrollBar:vertical {
                border: none;
                background: #f1f3f5;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #ced4da;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover { background: #adb5bd; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)
        right_layout.addWidget(self.log_text)
        
        content_layout.addWidget(right_panel, 65)
        
        container_layout.addLayout(content_layout)
        main_layout.addWidget(self.container)
        
        # Add resize handle (QSizeGrip) to the bottom right corner of the window
        self.size_grip = QSizeGrip(self)
        self.size_grip.setFixedSize(20, 20)
        
        self.log_redirector.set_widget(self.log_text)
        self._log("System initialized. Ready to launch.")

    def _on_provider_changed(self, index):
        key = self.provider_combo.itemData(index)
        provider = PROVIDERS.get(key, {})
        self.model_combo.clear()
        for model in provider.get('models', []):
            self.model_combo.addItem(model)
        self.base_url_input.setText(provider.get('base_url', ''))

    def _toggle_key_visibility(self):
        if self.api_key_input.echoMode() == QLineEdit.Password:
            self.api_key_input.setEchoMode(QLineEdit.Normal)
            self.show_key_btn.setText('🙈')
        else:
            self.api_key_input.setEchoMode(QLineEdit.Password)
            self.show_key_btn.setText('👁')

    def _load_config(self):
        if not os.path.exists(CONFIG_PATH):
            return
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception:
            return

        provider_key = config.get('provider', 'minimax')
        for i in range(self.provider_combo.count()):
            if self.provider_combo.itemData(i) == provider_key:
                self.provider_combo.setCurrentIndex(i)
                break

        self.api_key_input.setText(config.get('api_key', ''))
        self.base_url_input.setText(config.get('base_url', ''))
        model = config.get('model', config.get('vision_model', ''))
        if model:
            self.model_combo.setCurrentText(model)
        self._log('Configuration loaded.')

    def _save_config(self):
        config = {
            'provider': self.provider_combo.currentData(),
            'api_key': self.api_key_input.text().strip(),
            'model': self.model_combo.currentText().strip(),
            'base_url': self.base_url_input.text().strip()
        }
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            self._log('[INFO] API Configuration saved.')
            QMessageBox.information(self, 'Success', 'Configuration saved successfully.')
        except Exception as e:
            self._log(f'[ERROR] Save failed: {e}')
            QMessageBox.warning(self, 'Error', str(e))

    def _run_whiteboard(self):
        self._save_config()
        if not self.api_key_input.text().strip():
            QMessageBox.warning(self, 'Warning', 'API Key cannot be empty.')
            return

        self._log('-' * 40)
        self._log('>> LAUNCHING AI WHITEBOARD ENGINE...')
        self._log('-' * 40)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        threading.Thread(target=self._run_process, daemon=True).start()

    def _run_process(self):
        try:
            if getattr(sys, 'frozen', False):
                cmd = [sys.executable, '--run-core']
            else:
                main_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')
                cmd = [sys.executable, main_py]

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.log_redirector.log_signal.emit(line.rstrip())
            
            self.process.stdout.close()
            self.process.wait()
            exit_code = self.process.returncode
            self.log_redirector.log_signal.emit(f'>> Process exited with code {exit_code}')

        except Exception as e:
            self.log_redirector.log_signal.emit(f'[ERROR] Failed to launch: {e}')
        finally:
            self.process = None
            QTimer.singleShot(0, self._reset_buttons)

    def _reset_buttons(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _stop_whiteboard(self):
        if self.process:
            self._log('>> Terminating process...')
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self._log('>> Terminated.')
            self._reset_buttons()

    def _log(self, text):
        timestamp = time.strftime('%H:%M:%S')
        self.log_text.append(f'[{timestamp}] {text}')

    def _save_log_to_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存日志", "", "Text Files (*.txt);;All Files (*)")
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "成功", "日志已保存！")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存失败: {e}")

    def closeEvent(self, event):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.size_grip.move(self.width() - 20, self.height() - 20)

def main():
    # 启用高分屏支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = LauncherWindowV2()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--run-core':
        sys.argv.pop(1)
        import main as main_engine
        main_engine.main()
    else:
        main()
