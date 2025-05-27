import requests
import json
from dotenv import load_dotenv
import os
InternLM_api_key = os.getenv("InternLM", load_dotenv())

url = 'https://chat.intern-ai.org.cn/api/v1/chat/completions'
header = {
    'Content-Type':'application/json',
    "Authorization":"Bearer "+InternLM_api_key,
}
data = {
    "model": "internlm3-latest",  
    "messages": [{
        "role": "user",
        "content": "你好~"
    }],
    "n": 1,
    "temperature": 0.8,
    "top_p": 0.9
}

res = requests.post(url, headers=header, data=json.dumps(data))
print(res.status_code)
print(res.json())
print(res.json()["choices"][0]['message']["content"])