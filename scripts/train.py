"""
训练脚本
"""
import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cropping_net import CroppingNet
from core.dataset import CroppingDataset
from core.loss import CroppingLoss
from core.metrics import CroppingMetrics


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config):
    """构建模型"""
    model_cfg = config['model']
    model = CroppingNet(
        visual_encoder_name=model_cfg.get('visual_encoder', 'clip_vit_b32'),
        pretrained=model_cfg.get('pretrained', True),
        freeze_encoder=model_cfg.get('freeze_encoder', False),
        feature_dim=model_cfg.get('feature_dim', 512),
        text_dim=model_cfg.get('text_dim', 512),
        fusion_type=model_cfg.get('fusion_type', 'cross_attention'),
        roi_size=model_cfg.get('roi_align_size', 7),
        num_heads=model_cfg.get('num_attention_heads', 8),
        num_fusion_layers=model_cfg.get('num_fusion_layers', 2),
        dropout=model_cfg.get('dropout', 0.3),
    )
    return model


def build_optimizer(model, config):
    """构建优化器"""
    train_cfg = config['training']
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.get('learning_rate', 1e-4),
        weight_decay=train_cfg.get('weight_decay', 1e-4),
    )
    return optimizer


def build_scheduler(optimizer, config):
    """构建学习率调度器"""
    train_cfg = config['training']
    scheduler_type = train_cfg.get('lr_scheduler', 'cosine')
    num_epochs = train_cfg.get('num_epochs', 40)

    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    else:
        raise ValueError(f"不支持的调度器: {scheduler_type}")

    return scheduler


def collate_fn(batch):
    """
    自定义 collate 函数
    将每个样本的多个候选框展开为独立的样本
    
    数据输入：
    - 图像图：原始输入图像
    - 文字图：包含 semantic_caption（内容描述）和 emotion_caption（情感描述）
    
    训练策略：将情感单独作为特征进行训练
    """
    images = []
    all_rois = []
    all_scores = []
    semantic_captions = []  # 内容描述
    emotion_captions = []   # 情感描述（独立特征）
    image_ids = []

    for batch_idx, sample in enumerate(batch):
        image = sample['image']
        bboxes = sample['bboxes']
        scores = sample['scores']
        semantic_caption = sample['semantic_caption']  # 内容描述
        emotion_caption = sample['emotion_caption']    # 情感描述
        image_id = sample['image_id']

        images.append(image)
        semantic_captions.append(semantic_caption)
        emotion_captions.append(emotion_caption)
        image_ids.append(image_id)

        # 为每个候选框创建 ROI 条目
        num_bboxes = len(bboxes)
        for i, bbox in enumerate(bboxes):
            all_rois.append([batch_idx] + bbox.tolist())
            all_scores.append(scores[i].item())

    images = torch.stack([torch.tensor(img) for img in images])
    rois = torch.tensor(all_rois, dtype=torch.float32)
    scores = torch.tensor(all_scores, dtype=torch.float32)

    return {
        'images': images,
        'rois': rois,
        'scores': scores,
        'semantic_captions': semantic_captions,  # 内容描述
        'emotion_captions': emotion_captions,    # 情感描述（独立特征）
        'image_ids': image_ids,
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, config):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    train_cfg = config['training']
    gradient_clip = train_cfg.get('gradient_clip', 1.0)

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['images'].to(device)
        rois = batch['rois'].to(device)
        target_scores = batch['scores'].to(device)
        semantic_captions = batch['semantic_captions']  # 内容描述
        emotion_captions = batch['emotion_captions']    # 情感描述（独立特征）

        # 前向传播：输入图像图 + 文字图（包含内容描述和情感描述）
        pred_scores = model(images, rois, semantic_captions, emotion_captions)

        # 计算损失
        loss_dict = criterion(pred_scores, target_scores)
        loss = loss_dict['total_loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}',
        })

    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    metrics = CroppingMetrics(top_k_list=(4, 8))
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Validating')
    for batch in pbar:
        images = batch['images'].to(device)
        rois = batch['rois'].to(device)
        target_scores = batch['scores'].to(device)
        semantic_captions = batch['semantic_captions']  # 内容描述
        emotion_captions = batch['emotion_captions']    # 情感描述（独立特征）

        # 前向传播：输入图像图 + 文字图（包含内容描述和情感描述）
        pred_scores = model(images, rois, semantic_captions, emotion_captions)

        # 计算损失
        loss_dict = criterion(pred_scores, target_scores)
        total_loss += loss_dict['total_loss'].item()
        num_batches += 1

        # 更新指标（按图片分组）
        # 这里简化处理，直接传入所有分数
        metrics.update(pred_scores, target_scores)

    avg_loss = total_loss / num_batches
    metrics_dict = metrics.compute()

    return avg_loss, metrics_dict, str(metrics)


def main():
    parser = argparse.ArgumentParser(description='训练图像裁剪模型')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置随机种子
    seed = config['training'].get('seed', 42)
    set_seed(seed)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 构建模型
    model = build_model(config)
    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 构建数据集
    data_cfg = config['data']
    train_dataset = CroppingDataset(
        dataset_dir=data_cfg.get('dataset_dir', './data/GAIC'),
        split=data_cfg.get('train_set', 'train'),
        image_size=data_cfg.get('image_size', 224),
        augmentation=data_cfg.get('augmentation', True),
    )
    test_dataset = CroppingDataset(
        dataset_dir=data_cfg.get('dataset_dir', './data/GAIC'),
        split=data_cfg.get('test_set', 'test'),
        image_size=data_cfg.get('image_size', 224),
        augmentation=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training'].get('batch_size', 4),
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training'].get('batch_size', 4),
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 构建损失函数
    loss_cfg = config['loss']
    criterion = CroppingLoss(
        regression_type=loss_cfg.get('regression_type', 'smooth_l1'),
        regression_weight=loss_cfg.get('regression_weight', 1.0),
        ranking_weight=loss_cfg.get('ranking_weight', 0.5),
        contrastive_weight=loss_cfg.get('contrastive_weight', 0.1),
        smooth_l1_beta=loss_cfg.get('smooth_l1_beta', 1.0),
    )

    # 构建优化器和调度器
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # 训练循环
    num_epochs = config['training'].get('num_epochs', 40)
    save_dir = os.path.join(
        config['output'].get('save_dir', './checkpoints'),
        config['output'].get('experiment_name', 'exp01'),
    )
    os.makedirs(save_dir, exist_ok=True)

    best_metric = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, config)
        print(f"训练损失: {train_loss:.4f}")

        # 验证
        if (epoch + 1) % config['evaluation'].get('eval_interval', 1) == 0:
            val_loss, val_metrics, val_str = validate(model, test_loader, criterion, device)
            print(f"验证损失: {val_loss:.4f}")
            print(f"验证指标: {val_str}")

            # 保存最佳模型
            metric_value = val_metrics.get(config['evaluation'].get('metric', 'srcc'), 0.0)
            if metric_value > best_metric:
                best_metric = metric_value
                best_epoch = epoch
                if config['evaluation'].get('save_best', True):
                    save_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metric': metric_value,
                    }, save_path)
                    print(f"保存最佳模型 (metric={metric_value:.4f})")

        # 更新学习率
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(metric_value)
        else:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {cur_lr:.6f}")

    print(f"\n训练完成！最佳指标: {best_metric:.4f} (epoch {best_epoch + 1})")
    print(f"模型保存在: {save_dir}")


if __name__ == '__main__':
    main()
