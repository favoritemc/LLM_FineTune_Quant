---
license: cc-by-nc-4.0
library_name: transformers
---
# Model Details

MobileLLM is introduced: "[MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases](https://arxiv.org/abs/2402.14905)", published in ICML 2024.

**Model Developer**: Meta

**Model Architecture**: MobileLLM is an auto-regressive language model leveraging an optimized transformer architecture, specifically engineered for on-device applications with constrained resources.
MobileLLM integrated several key techniques including: (1) SwiGLU activation function, (2) deep and thin architectures, (3) embedding sharing, (4) grouped-query attention. MobileLLM-125M/350M attains a remarkable 2.7%/4.3% accuracy boost over preceding 125M/350M SoTA models on zero-shot commonsense reasoning tasks. In our updated version, we further demonstrate that our design philosophy scales effectively to larger models, with SoTA results for MobileLLM-600M/1B/1.5B.

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/660f893bae89429c07a32cdb/ahtsJXC5vBVIdmsMQDNHv.jpeg)

| | # Layers | # Attnetion Heads | # KV Heads | Token Dimension | Params | 
| --- | --- | --- | --- | --- | --- | 
| MobileLLM-125M |  30 | 9  | 3 | 576  | 124.6M |
| MobileLLM-350M |  32 | 15 | 5 | 960  | 345.3M |
| MobileLLM-600M |  40 | 18 | 6 | 1152 | 603.1M |
| MobileLLM-1B   |  54 | 20 | 5 | 1280 | 1.01B  |
| MobileLLM-1.5B |  54 | 25 | 5 | 1600 | 1.51B  |

| | Training Data | Input modalities | Output modalities | Context Length | GQA | Shared Embeddings | Token count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileLLM-125M | Publicly available online data. | Text | Text | 2k | Yes | Yes | 1T tokens |
| MobileLLM-350M | Publicly available online data. | Text | Text | 2k | Yes | Yes | 1T tokens |
| MobileLLM-600M | Publicly available online data. | Text | Text | 2k | Yes | Yes | 1T tokens |
| MobileLLM-1B   | Publicly available online data. | Text | Text | 2k | Yes | Yes | 1T tokens |
| MobileLLM-1.5B | Publicly available online data. | Text | Text | 2k | Yes | Yes | 1T tokens |


# How to use
We are providing 2 ways to run the model:

[HuggingFace](#huggingface)

[MobileLLM codebase](#mobilellm-codebase)

## HuggingFace
To load the pretrained model for further finetuning or evaluation:
```bash
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/MobileLLM-125M", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("facebook/MobileLLM-125M", trust_remote_code=True)
```
Note that the default tokenizer does not contain special tokens. For example you can use:
```bash
tokenizer.add_special_tokens(
    {
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
    }
)
```
## MobileLLM codebase
We provide the pretraining code in https://github.com/facebookresearch/MobileLLM

```bash
> git clone https://github.com/facebookresearch/MobileLLM
> pip install -r requirement.txt

# data pre-process and specify the data path in pretrain.sh
# run pretraining
> bash pretrain.sh 
```
We also provide evaluation script for calculating ppl of wikitext-2 test split:
```bash
> bash eval.sh
```

You can find more details in the GitHub repo.

# Training cost 
It takes the following number of days to train MobileLLM on 1T tokens using 32 NVIDIA A100 80G GPUs.
| 125M | 350M | 600M | 1B | 1.5B | 
| --- | --- | --- | --- | --- |
| ~3 days| ~6 days| ~8 days | ~12 days | ~18 days |


# Evaluation
We evaluate the pretrained MobileLLM models on Zero-shot Common Sense Reasoning tasks

## MobileLLM-125M

| model | boolq | piqa | siqa | hellaswag | winogrande | arc_easy | arc_challenge | obqa | avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OPT-125M | 41.3 | 25.2 | 57.5 | 62.0 | 41.9 | 31.1 | 31.2 | 50.8 | 42.6 |
| GPT-neo-125M | 40.7 | 24.8 | 61.3 | 62.5 | 41.9 | 29.7 | 31.6 | 50.7 | 42.9 |
| Pythia-160M | 40.0 | 25.3 | 59.5 | 62.0 | 41.5 | 29.9 | 31.2 | 50.9 | 42.5 |
| **MobileLLM-125M** | 43.9 | 27.1 | 60.2 | 65.3 | 42.4 | 38.9 | 39.5 | 53.1 | **46.3** |
| **MobileLLM-LS-125M** | 45.8 | 28.7 | 60.4 | 65.7 | 42.9 | 39.5 | 41.1 | 52.1 | **47.0** |

## MobileLLM-350M

| model | boolq | piqa | siqa | hellaswag | winogrande | arc_easy | arc_challenge | obqa | avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OPT-350M | 41.9 | 25.7 | 54.0 | 64.8 | 42.6 | 36.2 | 33.3 | 52.4 | 43.9 |
| Pythia-410M | 47.1 | 30.3 | 55.3 | 67.2 | 43.1 | 40.1 | 36.2 | 53.4 | 46.6 |
| **MobileLLM-350M** | 53.8 | 33.5 | 62.4 | 68.6 | 44.7 | 49.6 | 40.0 | 57.6 | **51.3** |
| **MobileLLM-LS-350M** | 54.4 | 32.5 | 62.8 | 69.8 | 44.1 | 50.6 | 45.8 | 57.2 | **52.1** | 

## MobileLLM-600M

| model | boolq | piqa | siqa | hellaswag | winogrande | arc_easy | arc_challenge | obqa | avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen1.5-500M | 54.7 | 32.1 | 46.9 | 68.9 | 46.0 |  48.8 | 37.7 | 55.0 | 48.8 | 
| BLOOM-560M | 43.7 | 27.5 | 53.7 | 65.1 | 42.5 | 36.5 | 32.6 | 52.2 | 44.2 | 
| MobiLlama-800M | 52.0 | 31.7 | 54.6 | 73.0 |  43.3 | 52.3 | 42.5 | 56.3 | 50.7 | 
| **MobileLLM-600M** | 58.1 |  35.8 |  61.0 |  72.3 | 44.9 | 55.9 |  47.9 |  58.6 | **54.3** |  

## MobileLLM-1B

| model | boolq | piqa | siqa | hellaswag | winogrande | arc_easy | arc_challenge | obqa | avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pythia-1B | 49.9 | 30.4 | 58.7 | 69.2 | 43.3 | 47.4 | 38.6 | 52.2 | 48.7 | 
| MobiLlama-1B | 59.7 | 38.4 | 59.2 | 74.5 | 44.9 | 62.0 | 43.7 | 59.0 | 55.2 | 
| Falcon-1B | 59.5 | 38.4 | 63.9 | 74.6 |  44.6 | 62.9 |  45.6 | 60.9 | 56.3 | 
| BLOOM-1.1B | 47.6 | 27.3 | 58.6 | 67.0 | 42.4 | 42.2 | 36.6 | 53.8 | 46.9 | 
| TinyLlama-1.1B | 59.2 | 37.1 | 58.1 | 72.9 | 43.9 | 59.1 | 44.7 | 58.8 | 54.2 | 
| **MobileLLM-1B** | 63.0 |  39.0 |  66.7 |  74.4 | 45.0 |  61.4 | 46.8 | 62.3 | **57.3** |  

## MobileLLM-1.5B

| model | boolq | piqa | siqa | hellaswag | winogrande | arc_easy | arc_challenge | obqa | avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-neo-1.3B | 51.3 | 33.0 | 61.8 | 70.9 | 43.7 | 48.6 | 41.2 | 54.5 | 50.6 | 
| OPT-1.3B | 54.4 | 31.7 | 58.4 | 71.5 | 44.7 | 53.7 | 44.6 | 59.1 | 52.3 | 
| BLOOM-1.7B | 50.9 | 31.2 | 61.7 | 70.0 | 43.2 | 47.2 | 36.2 | 56.1 | 49.6 | 
| Qwen1.5-1.8B | 61.1 | 36.5 | 68.3 | 74.1 | 47.2 |  60.4 | 42.9 | 61.2 | 56.5 | 
| GPT-neo-2.7B | 55.8 | 34.3 | 62.4 | 72.9 | 43.6 | 55.6 | 40.0 | 57.9 | 52.8 | 
| OPT-2.7B | 56.6 | 34.6 | 61.8 | 74.5 | 45.6 | 60.2 | 48.2 | 59.6 | 55.1 | 
| Pythia-2.8B | 59.4 | 38.9 | 66.1 |  73.8 | 44.5 | 59.6 | 45.0 | 59.4 | 55.8 | 
| BLOOM-3B | 55.1 | 33.6 | 62.1 | 70.5 | 43.2 | 53.9 | 41.6 | 58.2 | 52.3 | 
| **MobileLLM-1.5B** | 67.5 |  40.9 |  65.7 | 74.8 |  46.4 | 64.5 | 50.5 | 64.7 | **59.4** | 

# Citation

If you find our code useful for your research, please consider citing:
    
    @article{liu2024mobilellm,
        title={MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases},
        author={Liu, Zechun and Zhao, Changsheng and Iandola, Forrest and Lai, Chen and Tian, Yuandong and Fedorov, Igor and Xiong, Yunyang and Chang, Ernie and Shi, Yangyang and Krishnamoorthi, Raghuraman and others},
        journal={arXiv preprint arXiv:2402.14905},
        year={2024}
    }
    
# License

MobileLLM is CC-BY-NC 4.0 licensed as of now.