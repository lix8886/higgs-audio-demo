# higgs-audio easy-local-use 使用说明

## 参考音频文件

- 克隆的音频及对应的文本 xxx.wav + xxx.txt 放到 voice_prompts
- 可以设置音频 xxx 的 profile, 放到 voice_profile.json 中

## transcript/场景文本描述

1. 场景描述

   > Audio is recorded from a quiet room.

   > In this audio, the person is reading a blog post aloud. The content is informative and engaging, with the speaker using a clear, conversational tone to make the material feel more approachable. The pacing is moderate, allowing listeners to absorb the information, and the tone shifts slightly to emphasize key points. The speaker occasionally pauses for effect, ensuring each section flows smoothly, as they guide the listener through the post's main ideas.

2. 多人对话

   > [SPEAKER0] I can't believe you did that without even asking me first!
   > [SPEAKER1] Oh, come on! It wasn't a big deal, and I knew you would overreact like this.
   > [SPEAKER0] Overreact? You made a decision that affects both of us without even considering my opinion!
   > [SPEAKER1] Because I didn't have time to sit around waiting for you to make up your mind! Someone had to act.

3. 单人说话

   > 今天，我们来聊聊如何使用 Higgs Audio Generation 进行多模态对话。

   > 大家好，欢迎收听本期的跟李沐学 AI。今天沐哥在忙着洗数据，所以由我，希格斯主播代替他讲这期视频。
   > 今天我们要聊的是一个你绝对不能忽视的话题"多模态学习"。
   > 无论你是开发者，数据科学爱好者，还是只是对人工智能感兴趣的人都一定听说过这个词。它已经成为 AI 时代的一个研究热点。
   > 那么，问题来了，你真的了解多模态吗 你知道如何自己动手构建多模态大模型吗。

4. 实验性质

   > [music start] I will remember this, thought Ender, when I am defeated. To keep dignity, and give honor where it’s due, so that defeat is not disgrace. And I hope I don’t have to do it often. [music end]

   > Are you asking if I can hum a tune? Of course I can! [humming start] la la la la la [humming end] See?

## 测试代码

> conda create -n higgs-audio
> conda activate higgs-audio
> pip install -r requirements.txt
> 使用具体参考 main.ipynb

```python
from IPython.display import Audio
from generation import HiggsAudioModelClient

model_client = HiggsAudioModelClient(
    model_path="models/higgs-audio-v2-generation-3B-base",
    audio_tokenizer="models/higgs-audio-v2-tokenizer",
    semantic_model_or_path="models/hubert_base",
    device="cuda:4",
    use_static_kv_cache=True,
)
```

- example1

```python
model_client.create(
    "响应多样化处理：提供内容流式读取、编码转换、多种文本格式转换。",
    "Audio is recorded from a quiet room.",
)
Audio("generation.wav")
```

- example2

```python
transcript = """[SPEAKER0] I can't believe you did that without even asking me first!
[SPEAKER1] Oh, come on! It wasn't a big deal, and I knew you would overreact like this.
[SPEAKER0] Overreact? You made a decision that affects both of us without even considering my opinion!
[SPEAKER1] Because I didn't have time to sit around waiting for you to make up your mind! Someone had to act.
""".strip()
model_client.create(
    transcript,
    "Audio is recorded from a quiet room.",
)
Audio("generation.wav")
```

- example3

```python
transcript = """大家好，欢迎收听本期的跟李沐学 AI。今天沐哥在忙着洗数据，所以由我，希格斯主播代替他讲这期视频。
今天我们要聊的是一个你绝对不能忽视的话题"多模态学习"。
无论你是开发者，数据科学爱好者，还是只是对人工智能感兴趣的人都一定听说过这个词。它已经成为 AI 时代的一个研究热点。
那么，问题来了，你真的了解多模态吗 你知道如何自己动手构建多模态大模型吗。
""".strip()
model_client.create(
    transcript,
    "Happy, fast-paced",  # Audio is recorded from a quiet room.
    ref_audio="chadwick",
)
Audio("generation.wav", autoplay=True)
```

- example3

```python
transcript = """[SPEAKER0] I can't believe you did that without even asking me first!
[SPEAKER1] Oh, come on! It wasn't a big deal, and I knew you would overreact like this.
[SPEAKER0] Overreact? You made a decision that affects both of us without even considering my opinion!
[SPEAKER1] Because I didn't have time to sit around waiting for you to make up your mind! Someone had to act.
""".strip()
model_client.create(
    transcript,
    "Audio is recorded from a quiet room.",
    ref_audio="chadwick,profile:male_en_british",
)
Audio("generation.wav")
```

## 源代码链接

[https://github.com/boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio)
