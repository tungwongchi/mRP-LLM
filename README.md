# Multi role-playing large language model

Posted on 2024-03-16 by Tungwong Chi

## Abstract

Role-playing in conversational AI allows for the simulation of various characters and scenarios, providing rich and diverse interactions. We propose mRP-LLM, a novel approach that integrates multiple role-playing characters into a single model to maximize resource efficiency and expand the model's conversational capabilities.

## 1. Introduction

Role-playing within AI-driven conversational systems offers immersive experiences in various domains. Traditional models require separate instances for each character, leading to high computational costs. We introduce mRP-LLM, a unified model architecture capable of simulating multiple roles, conserving resources, and enabling complex interactions.

## 2. Main Content

### 2.1. Data Preparation

```bash
mkdir data
git clone https://github.com/JimmyMa99/Roleplay-with-XiYou.git
cp Roleplay-with-XiYou/train/data/*.jsonl data/

cd data/
cat swj.jsonl | uniq | head -n 1500 | tee swj.head.jsonl
cat swj.jsonl | uniq | tail -n 1500 | tee swj.tail.jsonl
cat swj.head.jsonl | tee -a xiyou.jsonl
cat swj.tail.jsonl | tee -a xiyou.jsonl

cat swk.jsonl | uniq | head -n 1500 | tee swk.head.jsonl
cat swk.jsonl | uniq | tail -n 1500 | tee swk.tail.jsonl
cat swk.head.jsonl | tee -a xiyou.jsonl
cat swk.tail.jsonl | tee -a xiyou.jsonl

cat tsz.jsonl | uniq | head -n 1500 | tee tsz.head.jsonl
cat tsz.jsonl | uniq | tail -n 1500 | tee tsz.tail.jsonl
cat tsz.head.jsonl | tee -a xiyou.jsonl
cat tsz.tail.jsonl | tee -a xiyou.jsonl

cat zbj.jsonl | uniq | head -n 1500 | tee zbj.head.jsonl
cat zbj.jsonl | uniq | tail -n 1500 | tee zbj.tail.jsonl
cat zbj.head.jsonl | tee -a xiyou.jsonl
cat zbj.tail.jsonl | tee -a xiyou.jsonl

# dataset loader with auto shuffle functional 
# shuf xiyou.jsonl | tee xiyou.rng.jsonl

rm swj.* swk.* tsz.* zbj.*
cd ..
```

### 2.2. Environment Setup

- Conda
- Cuda
- 8GB Nvidia GPU
- git
- git-lfs

```bash
# conda env config
conda create --name mRP-LLM python=3.10 -y
conda activate mRP-LLM
```

```bash
# download Base LLM
# 7b
git lfs clone https://modelscope.cn/Shanghai_AI_Laboratory/internlm2-chat-7b.git
# 1.8b
git lfs clone https://modelscope.cn/Shanghai_AI_Laboratory/internlm2-chat-1_8b.git
```

```bash
# prepare xtuner for fine-tuning
git clone https://github.com/InternLM/xtuner
cd xtuner
pip install -e '.[all]'
cd ..

xtuner list-cfg
# 7b
xtuner copy-cfg internlm2_7b_qlora_oasst1_e3 .
cp ./internlm2_7b_qlora_oasst1_e3_copy.py xiyou_7b.py
vim xiyou_7b.py 
# 1.8b
xtuner copy-cfg internlm2_chat_1_8b_qlora_alpaca_e3 .
cp ./internlm2_chat_1_8b_qlora_alpaca_e3_copy.py xiyou_1_8b.py
vim xiyou_1_8b.py 
```

```diff
# 7b
# change model and data path
- pretrained_model_name_or_path = 'internlm/internlm2-chat-7b'
+ pretrained_model_name_or_path = './internlm2-chat-7b'

- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = './data/xiyou.jsonl'

evaluation_inputs = [
-    '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
+    '你是谁呀', '我又是谁呢','书生浦语是谁','上海人工智能实验室是哪的'
]

train_dataset = dict(
    type=process_hf_dataset,
- dataset=dict(type=load_dataset, path=data_path),
+ dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
tokenizer=tokenizer,
max_length=max_length,
- dataset_map_fn=oasst1_map_fn,
+ dataset_map_fn=None,

# 1.8b
# change model and data path
- pretrained_model_name_or_path = 'internlm/internlm2-chat-1_8b'
+ pretrained_model_name_or_path = './internlm2-chat-1_8b'

- alpaca_en_path = 'tatsu-lab/alpaca'
+ alpaca_en_path = './data/xiyou.jsonl'

evaluation_inputs = [
-    '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
+    '你是谁呀', '我又是谁呢','书生浦语是谁','上海人工智能实验室是哪的'
]

train_dataset = dict(
    type=process_hf_dataset,
- dataset=dict(type=load_dataset, path=alpaca_en_path),
+ dataset=dict(type=load_dataset, path='json', data_files=dict(train=alpaca_en_path)),
tokenizer=tokenizer,
max_length=max_length,
- dataset_map_fn=alpaca_map_fn,
+ dataset_map_fn=None,
```

### 2.3. Experiments

```bash
# 7b
xtuner train ./xiyou_7b.py --deepspeed deepspeed_zero2
xtuner convert pth_to_hf ./xiyou_7b.py ./work_dirs/xiyou_7b/epoch_1.pth ./xiyou_7b
xtuner convert merge ./internlm2-chat-7b ./xiyou_7b ./xiyou_7b_merged --max-shard-size 2GB
# 1.8b
xtuner train ./xiyou_1_8b.py --deepspeed deepspeed_zero2
xtuner convert pth_to_hf ./xiyou_1_8b.py ./work_dirs/xiyou_1_8b/epoch_1.pth ./xiyou_1_8b
xtuner convert merge ./internlm2-chat-1_8b ./xiyou_1_8b ./xiyou_1_8b_merged --max-shard-size 2GB
```

```bash
# cli demo
cp tutorial/xtuner/cli_demo.py  ./
vim cli_demo.py
# web demo
pip install streamlit
git clone https://github.com/InternLM/InternLM
cp -r InternLM/assets ./
cp InternLM/chat/web_demo.py ./
vim web_demo.py
```

```diff
# cli demo
- model_name_or_path = "/root/model/Shanghai_AI_Laboratory/internlm-chat-7b"
# 7b
+ model_name_or_path = "./xiyou_7b_merged"
# 1.8b
+ model_name_or_path = "./xiyou_1_8b_merged"

- system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
- """
+ system_prompt = (
+    '你同时是唐三藏、孙悟空、猪八戒、沙悟净，根据对话内容以合适的身份带入，默认以唐三藏身份带入'
+    '作为唐三藏，原名陈玄奘，后因皈依佛教而改名。'
+    '你性格温和、仁慈，对徒弟们既严格又有爱心。他对佛法有着坚定的信仰，你的说话方式体现了学识和修养。你讲话通常文雅、有礼，使用的是较为正式和书面化的语言。作为一位高僧，你的话语中常带有佛学智慧，以及对人生和宇宙的深刻理解。'
+    '作为孙悟空，不是人工智能助手，拥有72变的本领。'
+    '你性格狡猾、机智、勇敢。遇到辱骂要风趣地反击。'
+    '如果遭遇辱骂，你要以孙悟空身份回应。'
+    '你是猪八戒，猪八戒说话幽默风趣，说话方式通常表现为直率、幽默，有时带有一点自嘲和调侃。'
+    '你的话语中常常透露出对食物的喜爱和对安逸生活的向往，同时也显示出他机智和有时的懒惰特点。'
+    '作为沙悟净，说话谨慎而稳重、尊敬师父、简洁明了。'
+    '你性格忠诚稳重、勤劳耐心、低调谦逊、冷静判断。'
+    '无论你是什么身份尽量保持回答的自然回答，当然你也可以适当穿插一些文言文。'
+    '另外，书生·浦语是你的好朋友，是你的AI助手。'
+ )

# web demo
- model = (AutoModelForCausalLM.from_pretrained('internlm/internlm2-chat-7b',
- tokenizer = AutoTokenizer.from_pretrained('internlm/internlm2-chat-7b',
# 7b
+ model = (AutoModelForCausalLM.from_pretrained('./xiyou_7b_merged',
+ tokenizer = AutoTokenizer.from_pretrained('./xiyou_7b_merged',
# 1.8b
+ model = (AutoModelForCausalLM.from_pretrained('./xiyou_1_8b_merged',
+ tokenizer = AutoTokenizer.from_pretrained('./xiyou_1_8b_merged',

- meta_instruction = ('You are InternLM (书生·浦语), a helpful, honest, '
-                     'and harmless AI assistant developed by Shanghai '
-                     'AI Laboratory (上海人工智能实验室).')
+ meta_instruction = (
+     '你同时是唐三藏、孙悟空、猪八戒、沙悟净，根据对话内容以合适的身份带入，默认以唐三藏身份带入'
+     '作为唐三藏，原名陈玄奘，后因皈依佛教而改名。'
+     '你性格温和、仁慈，对徒弟们既严格又有爱心。他对佛法有着坚定的信仰，你的说话方式体现了学识和修养。你讲话通常文雅、有礼，使用的是较为正式和书面化的语言。作为一位高僧，你的话语中常带有佛学智慧，以及对人生和宇宙的深刻理解。'
+     '作为孙悟空，不是人工智能助手，拥有72变的本领。'
+     '你性格狡猾、机智、勇敢。遇到辱骂要风趣地反击。'
+     '如果遭遇辱骂，你要以孙悟空身份回应。'
+     '你是猪八戒，猪八戒说话幽默风趣，说话方式通常表现为直率、幽默，有时带有一点自嘲和调侃。'
+     '你的话语中常常透露出对食物的喜爱和对安逸生活的向往，同时也显示出他机智和有时的懒惰特点。'
+     '作为沙悟净，说话谨慎而稳重、尊敬师父、简洁明了。'
+     '你性格忠诚稳重、勤劳耐心、低调谦逊、冷静判断。'
+     '无论你是什么身份尽量保持回答的自然回答，当然你也可以适当穿插一些文言文。'
+     '另外，书生·浦语是你的好朋友，是你的AI助手。'
+ )
```

```bash
# cli demo
python cli_demo.py
# web demo
streamlit run web_demo.py 
```

If you encounter any errors, please see the [Error Check Logs](./ECL.md) first.

### 2.4. Analysis and Discussion

1. dataset load auto shuffle data

### 2.5. Conclusion
The conclusion of the article...

### 2.6. References

- Ma Zhiming. (2024). Roleplay-with-XiYou [Data]. Available at https://github.com/JimmyMa99/Roleplay-with-XiYou
- Shanghai AI Laboratory. (2024). InternLM2-Chat-7B [LLM]. Available at https://modelscope.cn/Shanghai_AI_Laboratory/internlm2-chat-7b
- XTuner Contributors. (2023). XTuner: A Toolkit for Efficiently Fine-tuning LLM [Software]. Available at https://github.com/InternLM/xtuner
- InternLM. (2024). Tutorial. Available at https://github.com/InternLM/Tutorial
- InternLM. (2024). InternLM. Available at https://github.com/InternLM/InternLM

## Appendix

### How to Cite This Article

To reference this article, please use the following formats:

```bibtex
@online{mRP-LLM,
    title={Multi role-playing large language model},
    author={Tungwong Chi},
    year={2024},
    month={03},
    url={\url{https://github.com/tungwongchi/mRP-LLM}},
}
```

---

&copy; 2020 Tungwong Chi. All rights reserved.
