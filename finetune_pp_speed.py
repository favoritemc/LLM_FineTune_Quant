import argparse
import os
import math
import tqdm.auto as tqdm
import json
import torch
from torch.utils.data import Dataset
import datasets
import transformers
from torch.cuda.amp import GradScaler, autocast
import logging

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,  # 设置日志输出的最低级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('training.log', mode='w')  # 同时输出到文件
    ]
)
logger = logging.getLogger()  # 获取日志对象

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def move_to_device(*x_list, device):
    if len(x_list) > 1:
        return tuple([x.to(device) for x in x_list])
    else:
        return x_list[0].to(device)

class DatasetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.dataset[idx]["input_ids"]),
            torch.LongTensor(self.dataset[idx]["input_ids"]),
        )


# From DeepSpeed
class RepeatingLoader:
    def __init__(self, loader):
        """Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.
        """
        self.loader = loader
        self.data_iter = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=r'LLM_Model/MobileLLM-125M')
    parser.add_argument("--dataset_path", type=str, default='Data_sample/UMLSE_Train_Tokenized')
    parser.add_argument("--save_dir", type=str, default='Fine_Tuning_Results/UMLSE_whole')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)  # 更大的梯度累积步骤
    parser.add_argument("--num_train_steps", type=int, default=1500)
    parser.add_argument("--save_interval", type=int, default=500)
    args = parser.parse_args()

    logger.info("Setup Data")
    dataset = datasets.load_from_disk(args.dataset_path)
    dataloader = RepeatingLoader(torch.utils.data.DataLoader(
        DatasetDataset(dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4  # 增加数据加载时的多线程数
    ))

    logger.info("Setup Model")
    num_layers = read_json((args.model_path + "/config.json"))["num_hidden_layers"]
    device_ids = list(range(torch.cuda.device_count()))  # 获取所有可用的 GPU
    device_map = {
        "model.embed_tokens": device_ids[0],
        "model.norm.weight": device_ids[-1],
        "lm_head": device_ids[-1],
    }

    # 为每个模型层分配不同的 GPU
    allocations = [
        device_ids[i] for i in
        sorted(list(range(len(device_ids))) * math.ceil(num_layers / len(device_ids)))
    ]
    for layer_i, device_id in enumerate(allocations):
        device_map[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.down_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.up_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.input_layernorm.weight"] = device_id
        device_map[f"model.layers.{layer_i}.post_attention_layernorm.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = device_id

    model = transformers.LlamaForCausalLM.from_pretrained(
        args.model_path,
        device_map=device_map,  # 分配每层的 GPU
    )

    # 启用梯度检查点
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False  # 禁用缓存，以便于训练

    # 启用混合精度训练
    scaler = GradScaler()  # 初始化混合精度训练的 scaler

    # 设置优化器
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 多 GPU 支持
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)  # 使用 DataParallel 进行多 GPU 训练

    # Train
    logger.info("Start training")
    generator = iter(dataloader)
    for step in tqdm.trange(args.num_train_steps):
        input_ids, labels = next(generator)

        # 确保输入数据被移动到正确的设备上
        input_ids = input_ids.to(model.device)  # model.device 会自动选择主 GPU（cuda:0）
        labels = labels.to(model.device)

        # 使用 autocast 自动混合精度
        with autocast():
            output = model(input_ids=input_ids, labels=labels)
            loss = output['loss']

        # 使用 scaler 进行梯度缩放和优化器步骤
        scaler.scale(loss).backward()  # 计算梯度
        scaler.step(opt)  # 更新优化器
        scaler.update()  # 更新 scaler

        if step % 1 == 0:
            logger.info(f"Loss={loss.item():.3f}")

        actual_step = step + 1
        if actual_step % args.gradient_accumulation_steps == 0:
            opt.zero_grad()

        if actual_step % args.save_interval and actual_step != args.num_train_steps:
            model.save_pretrained(
                os.path.join(args.save_dir), f"checkpoint-{actual_step}",
                max_shard_size="500MB",
            )

    model.save_pretrained(
        os.path.join(args.save_dir), f"checkpoint-final",
        max_shard_size="500MB",
    )


if __name__ == "__main__":
    main()
