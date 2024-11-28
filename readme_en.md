Here's the English version of your `README.md`:

---

# Fine-tuning LLAMA Model

This project is designed to guide Chinese users in fine-tuning Large Language Models (LLAMA) ([LLAMA](https://arxiv.org/abs/2302.13971)). The project integrates several frameworks ([Minimal LLaMA](https://github.com/zphang/minimal-llama), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [LMFlow](https://github.com/OptimalScale/LMFlow/tree/main/src/lmflow)), avoiding unnecessary encapsulations and ensuring the readability and usability of the code.

## Environment Setup

1. **Install dependencies**

   First, it is recommended to install PyTorch and the related CUDA version via `conda` to ensure compatibility between CUDA and PyTorch. Use the following command:

   ```bash
   conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
   ```

2. **Install other necessary dependencies**

   Next, install other necessary Python libraries, including `sentencepiece` and `transformers`. You can install them with the following command:

   ```bash
   pip install sentencepiece
   pip install transformers
   ```

   If you need to use `peft` related features, enter the `Python_Package` directory and first use `pip` to install the online packages to ensure all dependencies are installed correctly:

   ```bash
   pip install -e .
   ```

3. **Download LLAMA model**

   Go to the `LLAMA_Model` directory and download the LLAMA model parameters. You can either download the pretrained LLAMA model from Hugging Face (e.g., [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)) or use the official tool to process your custom LLAMA model:

   ```bash
   python convert_llama_weights_to_hf.py --input_model <path_to_llama_weights> --output_model <output_directory>
   ```

## Data Preparation

1. **Prepare the dataset**

   Go to the `Data_sample` directory and prepare your dataset according to the provided example. The dataset should be pre-tokenized. If your dataset is in JSON format, make sure it can be loaded by the `datasets` library and is formatted for fine-tuning.

## Training

1. **Choose a fine-tuning script**

   Based on your needs, choose a fine-tuning script. If you want to fine-tune all model parameters, use `finetune_pp.py`. If you only want to fine-tune a subset of parameters (e.g., using LoRA technique), use `finetune_pp_peft.py`.

2. **Modify training parameters**

   Modify the relevant parameters in the script to specify the correct GPU and set other training parameters based on your hardware and task. Here is an example of how to run the training:

   ```bash
   python finetune_pp.py --model_path <LLAMA_model_path> --dataset_path <dataset_path> --save_dir <save_model_path> --num_train_steps 1500
   ```

   **Note**:
   - `finetune_pp.py` does not support multi-GPU acceleration, making it suitable for debugging and small-scale training.

### Start TensorBoard

During training, we recommend using TensorBoard for visualization to monitor metrics like loss and learning rate in real time.

1. **Start TensorBoard**:

   ```bash
   tensorboard --logdir=tensorboard_logs
   ```

2. **View training progress**:

   Open your browser and visit `http://localhost:6006` to view the training progress and visualizations.

## License

This project is licensed under the MIT License. For more information, please refer to the [LICENSE](LICENSE) file.

---

This version maintains all the original instructions while providing English translations for ease of understanding. If you need further adjustments, feel free to ask!