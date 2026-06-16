"""
智能教学白板 — 启动器前端
功能：配置 API、启动白板、查看控制台日志
"""

import sys
import os
import json
import time
import threading
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QGroupBox, QFormLayout,
    QComboBox, QFrame, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon


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
            # 自动滚动到底部
            scrollbar = self._log_widget.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())


class LauncherWindow(QWidget):
    """启动器主窗口"""

    def __init__(self):
        super().__init__()
        self.process = None
        self.log_redirector = LogRedirector()
        self._init_ui()
        self._load_config()

    def _init_ui(self):
        self.setWindowTitle('智能教学白板 — 启动器')
        self.resize(1600, 900)
        self.setMinimumSize(900, 500)

        # 深色主题
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 18px;
                font-weight: bold;
                font-size: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #89b4fa;
            }
            QLineEdit, QComboBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 6px 10px;
                color: #cdd6f4;
                font-size: 15px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #89b4fa;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                border: none;
                width: 30px;
                padding-right: 8px;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0;
                height: 0;
                border-left: 7px solid transparent;
                border-right: 7px solid transparent;
                border-top: 9px solid #cdd6f4;
            }
            QComboBox QAbstractItemView {
                background-color: #313244;
                color: #cdd6f4;
                selection-background-color: #45475a;
                border: 1px solid #45475a;
            }
            QPushButton {
                border-radius: 6px;
                padding: 8px 20px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton#runBtn {
                background-color: #a6e3a1;
                color: #1e1e2e;
            }
            QPushButton#runBtn:hover {
                background-color: #94e2d5;
            }
            QPushButton#runBtn:pressed {
                background-color: #74c7ec;
            }
            QPushButton#stopBtn {
                background-color: #f38ba8;
                color: #1e1e2e;
            }
            QPushButton#stopBtn:hover {
                background-color: #eba0ac;
            }
            QPushButton#saveBtn {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QPushButton#saveBtn:hover {
                background-color: #74c7ec;
            }
            QTextEdit {
                background-color: #11111b;
                border: 1px solid #45475a;
                border-radius: 6px;
                color: #a6adc8;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                padding: 6px;
            }
            QLabel {
                font-size: 14px;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # ===== 标题 =====
        title = QLabel('🎓 智能教学白板')
        title.setFont(QFont('Microsoft YaHei', 32, QFont.Bold))
        title.setStyleSheet('color: #89b4fa; margin-bottom: 4px; font-size: 32px;')
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        subtitle = QLabel('手势控制 · 屏幕批注 · AI 辅助分析')
        subtitle.setStyleSheet('color: #6c7086; font-size: 16px; margin-bottom: 8px;')
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)

        # ===== API 配置区 =====
        api_group = QGroupBox('API 配置')
        api_layout = QFormLayout(api_group)
        api_layout.setSpacing(8)
        api_layout.setContentsMargins(12, 20, 12, 12)

        # 服务商选择
        self.provider_combo = QComboBox()
        for key, val in PROVIDERS.items():
            self.provider_combo.addItem(val['name'], key)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        api_layout.addRow('服务商:', self.provider_combo)

        # API Key
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText('输入 API Key...')
        self.api_key_input.setEchoMode(QLineEdit.Password)

        # 显示/隐藏 Key 按钮
        key_layout = QHBoxLayout()
        self.show_key_btn = QPushButton('👁 显示')
        self.show_key_btn.setFixedWidth(60)
        self.show_key_btn.setStyleSheet('background-color: #45475a; color: #cdd6f4; font-size: 13px; padding: 4px;')
        self.show_key_btn.clicked.connect(self._toggle_key_visibility)
        key_layout.addWidget(self.api_key_input)
        key_layout.addWidget(self.show_key_btn)
        api_layout.addRow('API Key:', key_layout)

        # 模型选择
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        api_layout.addRow('模型:', self.model_combo)

        # Base URL
        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText('https://api.minimaxi.com/v1')
        api_layout.addRow('Base URL:', self.base_url_input)

        main_layout.addWidget(api_group)

        # ===== 按钮区 =====
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.save_btn = QPushButton('💾 保存配置')
        self.save_btn.setObjectName('saveBtn')
        self.save_btn.clicked.connect(self._save_config)

        self.run_btn = QPushButton('▶ 启动白板')
        self.run_btn.setObjectName('runBtn')
        self.run_btn.clicked.connect(self._run_whiteboard)

        self.stop_btn = QPushButton('■ 停止')
        self.stop_btn.setObjectName('stopBtn')
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_whiteboard)

        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.stop_btn)
        main_layout.addLayout(btn_layout)

        # ===== 控制台日志区 =====
        console_group = QGroupBox('控制台输出')
        console_layout = QVBoxLayout(console_group)
        console_layout.setContentsMargins(12, 20, 12, 12)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(160)
        console_layout.addWidget(self.log_text)

        # 清空按钮
        clear_layout = QHBoxLayout()
        clear_layout.addStretch()
        self.clear_btn = QPushButton('清空日志')
        self.clear_btn.setStyleSheet('background-color: #45475a; color: #cdd6f4; font-size: 13px; padding: 4px 12px;')
        self.clear_btn.clicked.connect(self.log_text.clear)
        clear_layout.addWidget(self.clear_btn)
        console_layout.addLayout(clear_layout)

        main_layout.addWidget(console_group, 1)  # stretch=1 让控制台区域随窗口扩展

        # 设置日志重定向
        self.log_redirector.set_widget(self.log_text)

    def _on_provider_changed(self, index):
        """切换服务商时更新模型列表和 base_url"""
        key = self.provider_combo.itemData(index)
        provider = PROVIDERS.get(key, {})
        
        # 更新模型列表
        self.model_combo.clear()
        for model in provider.get('models', []):
            self.model_combo.addItem(model)
        
        # 更新 base_url
        self.base_url_input.setText(provider.get('base_url', ''))

    def _toggle_key_visibility(self):
        """切换 API Key 显示/隐藏"""
        if self.api_key_input.echoMode() == QLineEdit.Password:
            self.api_key_input.setEchoMode(QLineEdit.Normal)
            self.show_key_btn.setText('🙈 隐藏')
        else:
            self.api_key_input.setEchoMode(QLineEdit.Password)
            self.show_key_btn.setText('👁 显示')

    def _load_config(self):
        """从 api_config.json 加载配置"""
        if not os.path.exists(CONFIG_PATH):
            return

        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception:
            return

        # 匹配服务商
        provider_key = config.get('provider', 'minimax')
        for i in range(self.provider_combo.count()):
            if self.provider_combo.itemData(i) == provider_key:
                self.provider_combo.setCurrentIndex(i)
                break

        # 填充字段
        self.api_key_input.setText(config.get('api_key', ''))
        self.base_url_input.setText(config.get('base_url', ''))

        model = config.get('model', config.get('vision_model', ''))
        if model:
            self.model_combo.setCurrentText(model)

        self._log(f'配置已加载: {CONFIG_PATH}')

    def _save_config(self):
        """保存配置到 api_config.json"""
        config = {
            'provider': self.provider_combo.currentData(),
            'api_key': self.api_key_input.text().strip(),
            'model': self.model_combo.currentText().strip(),
            'base_url': self.base_url_input.text().strip()
        }

        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            self._log('✅ 配置已保存')
            QMessageBox.information(self, '保存成功', 'API 配置已保存到 api_config.json')
        except Exception as e:
            self._log(f'❌ 保存失败: {e}')
            QMessageBox.warning(self, '保存失败', str(e))

    def _run_whiteboard(self):
        """启动白板程序"""
        # 先保存配置
        self._save_config()

        # 检查 API Key
        if not self.api_key_input.text().strip():
            QMessageBox.warning(self, '提示', '请先填写 API Key')
            return

        self._log('=' * 50)
        self._log('🚀 启动白板程序...')
        self._log('=' * 50)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # 在后台线程启动子进程
        main_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')
        threading.Thread(target=self._run_process, args=(main_py,), daemon=True).start()

    def _run_process(self, script_path):
        """后台运行白板进程"""
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-u', script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.path.dirname(script_path)
            )

            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.log_redirector.log_signal.emit(line.rstrip())
            
            self.process.stdout.close()
            self.process.wait()

            exit_code = self.process.returncode
            self.log_redirector.log_signal.emit(f'\n进程退出，返回码: {exit_code}')

        except Exception as e:
            self.log_redirector.log_signal.emit(f'\n启动失败: {e}')
        finally:
            self.process = None
            # 恢复按钮状态（通过信号）
            QTimer.singleShot(0, self._reset_buttons)

    def _reset_buttons(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _stop_whiteboard(self):
        """停止白板程序"""
        if self.process:
            self._log('⏹ 正在停止...')
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self._log('已停止')
            self._reset_buttons()

    def _log(self, text):
        """添加日志"""
        timestamp = time.strftime('%H:%M:%S')
        self.log_text.append(f'[{timestamp}] {text}')

    def closeEvent(self, event):
        """关闭窗口时停止子进程"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = LauncherWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
