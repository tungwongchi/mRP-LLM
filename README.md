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

shuf xiyou.jsonl | tee xiyou.rng.jsonl

rm swj.* swk.* tsz.* zbj.*
cd ..
```

### 2.2. Environment Setup

- Conda
- Cuda
- 8GB Nvidia GPU

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
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
mv internlm_chat_7b_qlora_oasst1_e3_copy.py xiyou_7b.py
mv internlm_chat_7b_qlora_oasst1_e3_copy.py xiyou_7b_rng.py
vim xiyou_7b.py xiyou_7b_rng.py
```

```diff
# change model and data path
- pretrained_model_name_or_path = 'internlm/internlm2-chat-7b'
+ pretrained_model_name_or_path = './internlm2-chat-7b'

- data_path = 'timdettmers/openassistant-guanaco'
# xiyou_7b.py
+ data_path = './data/xiyou.jsonl'
# xiyou_7b_rng.py
+ data_path = './data/xiyou.rng.jsonl'
```

### 2.3. Experiments


```bash
xtuner train ./xiyou_7b.py --deepspeed deepspeed_zero2
xtuner convert pth_to_hf ./xiyou_7b.py ./work_dirs/xiyou_7b/epoch_1.pth ./xiyou_7b
xtuner convert merge ./internlm2-chat-7b ./xiyou_7b ./xiyou_7b_merged --max-shard-size 2GB
xtuner chat ./xiyou_7b_merged --prompt-template internlm_chat
# shuffle data
xtuner train ./xiyou_7b_rng.py --deepspeed deepspeed_zero2
xtuner convert pth_to_hf ./xiyou_7b_rng.py ./work_dirs/xiyou_7b_rng/epoch_1.pth ./xiyou_7b_rng
xtuner convert merge ./internlm2-chat-7b ./xiyou_7b_rng ./xiyou_7b_rng_merged --max-shard-size 2GB
xtuner chat ./xiyou_7b_rng_merged --prompt-template internlm_chat
```

### 2.4. Analysis and Discussion
The analysis and discussion of the article...

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
@online{refTitl,
    title={title},
    author={Tungwong Chi},
    year={year},
    month={month},
    url={\url{https://url}},
}
```

---

&copy; 2020 Tungwong Chi. All rights reserved.
