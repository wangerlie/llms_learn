1. 为了使用HF的模型，需要现在HF登录账号获得token。

2. 为了方便下载模型，需要把资源网址设定为镜像网站，可以用`export HF_ENDPOINT=https://hf-mirror.com`

3. 飞书文档中的模型名称`internlm/InternVL2-1B`有误，应该是`OpenGVLab/InternVL3-1B`.

4. 运行下面脚本，将自动从HF下载模型并部署到本地
 ```python 
import lmdeploy
from lmdeploy import GenerationConfig
pipe = lmdeploy.pipeline("OpenGVLab/InternVL3-1B")
response = pipe(prompts=["Hi, pls intro yourself", "Shanghai is"],
                gen_config=GenerationConfig(max_new_tokens=1024,
                                            top_p=0.8,
                                            top_k=40,
                                            temperature=0.6))
print(response)
```

5. 模型回答如下：
```plain text
[Response(text="Hello! I'm an AI assistant whose name is InternVL, developed by Shanghai AI Lab and Tsinghua University. I'm here to help with a wide range of questions and tasks by providing information, answering problems, and assisting with various topics. How can I help you today?", generate_token_len=57, input_token_len=50, finish_reason='stop', token_ids=[9707, 0, 358, 2776, 458, 15235, 17847, 6693, 829, 374, 4414, 30698, 11, 7881, 553, 37047, 15235, 476, 1184, 1995, 389, 264, 3953, 12893, 315, 432, 11, 2666, 1910, 311, 2548, 0], logprobs=None, logits=None, last_hidden_state=None, index=1)]
```

6. 单图

```python
from lmdeploy import pipeline, VisionConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
pipe = pipeline('OpenGVLab/InternVL3-1B')

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe((f'describe this image{IMAGE_TOKEN}', image))
print(response)
```

回复：

```plain text
Response(text='The image shows a tiger lying on a grassy field. The tiger has distinctive orange fur with black stripes, and it appears to be relaxed, with its front paws resting on the grass. The background is a lush green, suggesting a natural or zoo-like setting.', generate_token_len=54, input_token_len=1843, finish_reason='stop', token_ids=[785, 2168, 4933, 264, 51735, 20446, 389, 264, 16359, 88, 2070, 13, 576, 51735, ], logprobs=None, logits=None, last_hidden_state=None, index=0)
```

```
Response(text='The image shows a tiger lying on a grassy field. The tiger has distinctive orange fur with black stripes, and it appears to be relaxed, with its front paws resting on the grass. The background is a lush green, suggesting a natural habitat or a zoo enclosure.', generate_token_len=55, input_token_len=1842, finish_reason='stop',
```

7. 多图

```python
from lmdeploy import pipeline, VisionConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
pipe = pipeline('OpenGVLab/InternVL3-1B')

image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]
images = [load_image(img_url) for img_url in image_urls]
response = pipe(('describe these images', images))
print(response)
```

回复

```
Response(text='The image shows a park setting with a bench in the foreground. The bench is made of metal and has a curved backrest. It is situated on a concrete platform in a grassy area. In the background, there are several parked cars along a road, and a few trees provide shade. The scene appears to be a quiet, sunny day, with the park area looking peaceful and well-maintained.', generate_token_len=82, input_token_len=3635, finish_reason='stop', token_ids=[785, 2168, 4933, 264, 6118, 6243, 448, 264, 13425, 304, 279, 39305, 13, 576, 13425, 374, 1865, 315, 9317, 323, 702, 264, 49164, 1182, 3927, 13, 1084, 374], logprobs=None, logits=None, last_hidden_state=None, index=0)
```

8. 多轮次对话

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

pipe = pipeline('OpenGVLab/InternVL3-1B',
                backend_config=TurbomindEngineConfig(session_len=8192))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.6)
sess = pipe.chat(('describe this image', image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat('What is the woman doing?', session=sess, gen_config=gen_config)
print(sess.response.text)
```

回复：

```
The image shows a person skiing on a snowy slope. They are wearing a red and white jacket, black pants, a striped beanie, sunglasses, and gloves. The skier is holding ski poles and appears to be in a relaxed stance. The background is a snowy landscape with a ski trail visible.
The woman is skiing on a snowy slope. She is holding ski poles and seems to be in a relaxed skiing posture.
```