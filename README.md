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
git lfs clone https://modelscope.cn/Shanghai_AI_Laboratory/internlm2-chat-7b.git
```

```bash
# prepare xtuner for fine-tuning
git clone https://github.com/InternLM/xtuner
cd xtuner
pip install -e '.[all]'

cd ..
xtuner list-cfg
xtuner copy-cfg internlm2_7b_qlora_oasst1_e3 .
cp ./internlm2_7b_qlora_oasst1_e3_copy.py xiyou_7b.py
vim xiyou_7b.py 
```

```diff
# change model and data path
- pretrained_model_name_or_path = 'internlm/internlm2-chat-7b'
+ pretrained_model_name_or_path = './internlm2-chat-7b'

- data_path = 'timdettmers/openassistant-guanaco'
# xiyou_7b.py
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
```

### 2.3. Experiments

```bash
xtuner train ./xiyou_7b.py --deepspeed deepspeed_zero2
xtuner convert pth_to_hf ./xiyou_7b.py ./work_dirs/xiyou_7b/epoch_1.pth ./xiyou_7b
xtuner convert merge ./internlm2-chat-7b ./xiyou_7b ./xiyou_7b_merged --max-shard-size 2GB
xtuner chat ./xiyou_7b_merged --prompt-template internlm_chat
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
