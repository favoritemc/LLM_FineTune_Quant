# 微调LLAMA模型

本项目旨在引导中文用户微调Large Language Model（[LLAMA](https://arxiv.org/abs/2302.13971)）。项目整合了多个框架（[Minimal LLaMA](https://github.com/zphang/minimal-llama)、[Alpaca](https://github.com/tatsu-lab/stanford_alpaca)、[LMFlow](https://github.com/OptimalScale/LMFlow/tree/main/src/lmflow)），避免不必要的封装，确保代码的可读性和易用性。

## 环境搭建

1. **安装依赖包**

   首先，建议通过`conda`安装PyTorch和相关的CUDA版本，这样可以确保CUDA与PyTorch版本的兼容性。可以使用以下命令：

   ```bash
   conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
   ```

2. **安装其他必要依赖**

   然后，安装其他必要的Python库，包括`sentencepiece`和`transformers`。可以通过以下命令安装：

   ```bash
   pip install sentencepiece
   pip install transformers
   ```

   如果你需要使用`peft`相关功能，可以进入`Python_Package`目录，并先使用`pip`安装线上包，确保所有依赖都能顺利安装：

   ```bash
   pip install -e .
   ```

3. **下载LLAMA模型**

   进入LLAMA_Model目录，下载LLAMA模型参数。你可以从Hugging Face下载预训练的LLAMA模型（如[decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)），或者使用官方工具处理自定义的LLAMA模型：

   ```bash
   python convert_llama_weights_to_hf.py --input_model <path_to_llama_weights> --output_model <output_directory>
   ```

## 数据处理

1. **准备数据集**

   进入`Data_sample`目录，并按照示例处理数据。数据集应该是已经tokenized（分词）好的。如果你的数据集格式是JSON，确保它能够被`datasets`库加载，并且符合微调所需的输入格式。

## 训练

1. **选择微调脚本**

   根据需求选择微调脚本。如果你想微调整个模型的参数，可以使用`finetune_pp.py`，如果只想微调部分参数（例如使用LoRA技术），则使用`finetune_pp_peft.py`。

2. **修改训练参数**

   修改脚本中的相关参数，确保指定正确的GPU，并设置适合你的硬件和任务的其他训练参数。以下是一个基本的启动命令：

   ```bash
   python finetune_pp.py --model_path <LLAMA模型路径> --dataset_path <数据集路径> --save_dir <保存模型路径> --num_train_steps 1500
   ```

   **注意**：
   - `finetune_pp.py`不支持多卡加速，适合用于调试和小规模训练。


### 启动TensorBoard

训练过程中，我们建议使用TensorBoard进行可视化，以便实时监控训练过程中的损失、学习率等指标。

1. **启动TensorBoard**：

   ```bash
   tensorboard --logdir=tensorboard_logs
   ```

2. **查看训练进度**：

   打开浏览器并访问`http://localhost:6006`，你将看到训练过程的可视化内容。

## 许可证

该项目发布在MIT许可证下，更多信息请参考[LICENSE](LICENSE)文件。

---