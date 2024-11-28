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

# 数据处理相关函数
def read_json(path):
    """读取 JSON 文件并返回内容"""
    with open(path, "r") as f:
        return json.load(f)


class DatasetDataset(torch.utils.data.Dataset):
    """ 自定义数据集类，用于加载并返回输入数据 """
    def __init__(self, dataset):
        """初始化数据集"""
        self.dataset = dataset

    def __len__(self):
        """返回数据集的大小"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """根据索引返回输入和标签"""
        return (
            torch.LongTensor(self.dataset[idx]["input_ids"]),
            torch.LongTensor(self.dataset[idx]["input_ids"]),
        )


class RepeatingLoader:
    """将 DataLoader 包装成无限循环的迭代器"""
    def __init__(self, loader):
        """
        初始化时，接收一个数据加载器。
        loader: 一个 PyTorch DataLoader 实例
        """
        self.loader = loader
        self.data_iter = iter(self.loader)

    def __iter__(self):
        """返回自身迭代器"""
        return self

    def __next__(self):
        """获取下一个批次，如果迭代结束则重置为第一个"""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch


# 模型配置与训练相关函数
def move_to_device(*x_list, device):
    """将所有传入的张量移动到指定的设备"""
    if len(x_list) > 1:
        return tuple([x.to(device) for x in x_list])
    else:
        return x_list[0].to(device)


def setup_model_and_device(args):
    """配置模型和设备，并返回模型和设备的映射"""
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

    return model, device_ids


def setup_optimizer(model, args):
    """设置优化器，返回优化器实例"""
    return torch.optim.AdamW(model.parameters(), lr=args.learning_rate)


def train_step(model, input_ids, labels, opt, scaler):
    """执行训练的一步，并返回损失"""
    # 使用 autocast 自动混合精度
    with autocast():
        output = model(input_ids=input_ids, labels=labels)
        loss = output['loss']

    # 使用 scaler 进行梯度缩放和优化器步骤
    scaler.scale(loss).backward()  # 计算梯度
    scaler.step(opt)  # 更新优化器
    scaler.update()  # 更新 scaler

    return loss


def log_train_status(step, loss, optimizer, actual_step, gradient_accumulation_steps, num_train_steps):
    """记录训练状态，包括损失、学习率和梯度累计等信息"""
    logger.info(f"Step {step}/{num_train_steps} - Loss={loss.item():.3f} - "
                f"LR={optimizer.param_groups[0]['lr']:.6f} - Grad Accumulation Step={actual_step % gradient_accumulation_steps}")


def save_checkpoint(model, save_dir, step, suffix=""):
    """保存模型的检查点"""
    model.save_pretrained(
        os.path.join(save_dir), f"checkpoint-{step}{suffix}",
        max_shard_size="500MB",
    )


# 主训练函数
def main():
    """主函数，进行训练过程"""
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=r'LLM_Model/MobileLLM-125M')
    parser.add_argument("--dataset_path", type=str, default='Data_sample/UMLSE_Train_Tokenized')
    parser.add_argument("--save_dir", type=str, default='Fine_Tuning_Results/UMLSE_whole')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_steps", type=int, default=1500)
    parser.add_argument("--save_interval", type=int, default=500)
    args = parser.parse_args()

    # 数据加载
    logger.info("Setup Data")
    dataset = datasets.load_from_disk(args.dataset_path)
    dataloader = RepeatingLoader(torch.utils.data.DataLoader(
        DatasetDataset(dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    ))

    # 配置模型和设备
    model, device_ids = setup_model_and_device(args)

    # 启用混合精度训练
    scaler = GradScaler()  # 初始化混合精度训练的 scaler

    # 设置优化器
    opt = setup_optimizer(model, args)

    # 多 GPU 支持
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # 训练开始
    logger.info("Start training")
    generator = iter(dataloader)
    for step in tqdm.trange(args.num_train_steps, desc="Training Progress", ncols=100):
        input_ids, labels = next(generator)

        # 确保输入数据被移动到正确的设备上
        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)

        # 执行训练步
        loss = train_step(model, input_ids, labels, opt, scaler)

        # 记录训练状态
        log_train_status(step, loss, opt, step + 1, args.gradient_accumulation_steps, args.num_train_steps)

        # 梯度累积步骤
        if (step + 1) % args.gradient_accumulation_steps == 0:
            opt.zero_grad()

        # 定期保存模型检查点
        if (step + 1) % args.save_interval == 0 and (step + 1) != args.num_train_steps:
            save_checkpoint(model, args.save_dir, step + 1)

    # 保存最终检查点
    save_checkpoint(model, args.save_dir, step + 1, suffix="-final")


if __name__ == "__main__":
    main()
