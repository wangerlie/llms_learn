from openai import OpenAI
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()
InternLM_api_key = os.getenv("InternLM")

# 初始化客户端
client = OpenAI(
    api_key=InternLM_api_key,  
    base_url="https://chat.intern-ai.org.cn/api/v1/",
)

# 构建包含3轮对话的上下文
messages = [
    # 第一轮对话
    {
        "role": "user",
        "content": "你好，请介绍一下你自己"
    },
    {
        "role": "assistant",
        "content": "你好！我是 InternVL，一个多模态AI助手。我可以理解和分析文本与图像，帮助您处理各种任务，包括图像描述、视觉问答、文档理解等。有什么我可以帮助您的吗？"
    },
    
    # 第二轮对话
    {
        "role": "user",
        "content": "你能处理哪些类型的图像任务？"
    },
    {
        "role": "assistant",
        "content": "我可以处理多种图像任务：\\n1. 图像描述：详细描述图片内容\\n2. 视觉问答：回答关于图片的问题\\n3. 图像对比：分析多张图片的异同\\n4. OCR识别：提取图片中的文字\\n5. 图表分析：理解图表、图形数据\\n6. 场景理解：识别图片中的物体、人物和场景"
    },
    
    # 第三轮对话 - 包含图像
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "请分析这两张图片，告诉我它们分别展示了什么产品，以及它们的主要特点"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://static.openxlab.org.cn/internvl/demo/visionpro.png"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://static.openxlab.org.cn/puyu/demo/000-2x.jpg"
                }
            }
        ]
    }
]

# 调用API获取最终回复
try:
    chat_rsp = client.chat.completions.create(
        model="internvl2.5-latest",
        messages=messages,
        n=1,
        stream=False,
        temperature=0.8,  # 控制回复的创造性
        max_tokens=500    # 限制回复长度
    )
    
    # 打印最终回复
    print("=== API 最终回复 ===")
    for choice in chat_rsp.choices:
        print(choice.message.content)
        
    # 可选：打印整个对话历史
    print("=== 完整对话历史 ===")
    for i, msg in enumerate(messages):
        print(f"轮次 {i//2 + 1}:")
        if msg["role"] == "user":
            if isinstance(msg["content"], str):
                print(f"用户: {msg['content']}")
            else:
                print(f"用户: {msg['content'][0]['text']}")
                print(f"      (包含 {len(msg['content'])-1} 张图片)")
        else:
            print(f"助手: {msg['content']}")
            
except Exception as e:
    print(f"API调用出错: {str(e)}")