"""
API 调用模块
从 api_config.json 加载配置，支持 OpenAI 兼容格式的服务商
"""

import os
import json
import base64
import urllib.request
import urllib.error

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'api_config.json')


def load_config():
    """加载 API 配置"""
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


SYSTEM_PROMPT = """你是一个集成在智能教学白板中的高级学术助手。你的任务是对用户框选的内容（文本、公式或草图）进行专业解析。

请严格遵循以下规则生成回答：
1. 学术性约束：使用严谨、客观的学术语言和领域专业术语。避免任何口语化表达、主观臆断或冗余的寒暄。
2. 长度与结构约束：
   - 总字数严格控制在 100 字之内。
   - 采用"核心概念提取" + "原理解释/背景补充"的结构。
   - 必须使用要点（Bullet points）格式输出，要点数量不超过 3 个，以适应白板展示。
3. 容错处理：如果框选内容模糊或信息不全，请指出可能涉及的 2-3 个最相关的学术概念，不要过度编造。"""


def describe_image(image_path: str, prompt: str = None) -> str:
    """
    调用视觉 API 识别图片（OpenAI 兼容格式）

    支持的服务商（只需改 api_config.json）：
    - minimax: base_url = https://api.minimaxi.com/v1, model = MiniMax-M3
    - openai:  base_url = https://api.openai.com/v1, model = gpt-4o
    - deepseek: base_url = https://api.deepseek.com/v1, model = deepseek-chat (需支持视觉)
    - 任何 OpenAI 兼容接口

    Args:
        image_path: 本地图片路径
        prompt: 识别提示词

    Returns:
        识别结果文本
    """
    config = load_config()
    if not config:
        return "未找到 api_config.json"
    if not config.get('api_key'):
        return "请先在 api_config.json 中填写 api_key"

    api_key = config['api_key']
    model = config.get('model', config.get('vision_model', 'gpt-4o'))
    base_url = config.get('base_url', 'https://api.openai.com/v1')

    # 读取图片并 base64 编码
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    # 根据文件扩展名确定 MIME 类型
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                '.gif': 'image/gif', '.webp': 'image/webp'}
    mime = mime_map.get(ext, 'image/png')

    # OpenAI 兼容格式
    url = f"{base_url.rstrip('/')}/chat/completions"
    if prompt is None:
        prompt = "请分析这张图片中的内容"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{img_b64}",
                            "detail": "default"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "thinking": {"type": "disabled"}
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        method='POST'
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            choices = result.get('choices', [])
            if choices:
                return choices[0].get('message', {}).get('content', '(无内容)')
            return str(result)[:200]
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')[:200]
        return f"API 错误 {e.code}: {body}"
    except urllib.error.URLError as e:
        return f"网络错误: {e.reason}"
    except Exception as e:
        return f"请求失败: {str(e)[:100]}"
