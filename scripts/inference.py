"""
推理脚本
"""
import os
import sys
import yaml
import argparse
import torch
import cv2
import numpy as np
from torchvision.transforms import functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cropping_net import CroppingNet


def load_model(model_path, config, device):
    """加载训练好的模型"""
    model_cfg = config['model']
    model = CroppingNet(
        visual_encoder_name=model_cfg.get('visual_encoder', 'clip_vit_b32'),
        pretrained=False,
        feature_dim=model_cfg.get('feature_dim', 512),
        text_dim=model_cfg.get('text_dim', 512),
        fusion_type=model_cfg.get('fusion_type', 'cross_attention'),
        roi_size=model_cfg.get('roi_align_size', 7),
        num_heads=model_cfg.get('num_attention_heads', 8),
        num_fusion_layers=model_cfg.get('num_fusion_layers', 2),
        dropout=model_cfg.get('dropout', 0.3),
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def preprocess_image(image_path, image_size=224, rgb_mean=(0.48145466, 0.4578275, 0.40821073),
                     rgb_std=(0.26862954, 0.26130258, 0.27577711)):
    """预处理图像"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]
    scale = image_size / min(h, w)
    new_h = int(round(h * scale / 32) * 32)
    new_w = int(round(w * scale / 32) * 32)
    image = cv2.resize(image, (new_w, new_h))

    image = image.astype(np.float32) / 255.0
    image = (image - np.array(rgb_mean)) / np.array(rgb_std)
    image = image.transpose(2, 0, 1)

    return torch.tensor(image).unsqueeze(0), (h, w, scale)


def generate_candidate_boxes(h, w, num_candidates=64):
    """生成候选裁剪框"""
    boxes = []
    # 生成不同尺度和位置的候选框
    scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    aspect_ratios = [0.75, 1.0, 1.33]

    for scale in scales:
        for ratio in aspect_ratios:
            box_h = h * scale
            box_w = box_h * ratio

            if box_h > h or box_w > w:
                continue

            # 在图像上均匀采样中心点
            for i in range(int(np.sqrt(num_candidates / (len(scales) * len(aspect_ratios)))) + 1):
                for j in range(int(np.sqrt(num_candidates / (len(scales) * len(aspect_ratios)))) + 1):
                    cx = (j + 0.5) * w / (int(np.sqrt(num_candidates / (len(scales) * len(aspect_ratios)))) + 1)
                    cy = (i + 0.5) * h / (int(np.sqrt(num_candidates / (len(scales) * len(aspect_ratios)))) + 1)

                    x1 = max(0, cx - box_w / 2)
                    y1 = max(0, cy - box_h / 2)
                    x2 = min(w, cx + box_w / 2)
                    y2 = min(h, cy + box_h / 2)

                    boxes.append([x1, y1, x2, y2])

    # 如果候选框太多，随机采样
    if len(boxes) > num_candidates:
        indices = np.random.choice(len(boxes), num_candidates, replace=False)
        boxes = [boxes[i] for i in indices]

    return boxes


@torch.no_grad()
def inference(model, image_path, caption="a photo", num_candidates=64, device='cuda'):
    """
    对单张图像进行推理

    Args:
        model: 训练好的模型
        image_path: 图像路径
        caption: 图像的情感描述
        num_candidates: 候选框数量
        device: 设备

    Returns:
        best_box: 最佳裁剪框坐标 [x1, y1, x2, y2]
        scores: 所有候选框的分数
        boxes: 所有候选框坐标
    """
    # 预处理
    image_tensor, (orig_h, orig_w, scale) = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    # 生成候选框
    boxes = generate_candidate_boxes(image_tensor.shape[2], image_tensor.shape[3], num_candidates)

    if len(boxes) == 0:
        raise ValueError("无法生成候选框")

    # 准备 ROI 输入
    rois = [[0] + box for box in boxes]
    rois = torch.tensor(rois, dtype=torch.float32).to(device)

    # 推理
    scores = model(image_tensor, rois, [caption] * len(boxes))

    # 找到最佳裁剪框
    best_idx = scores.argmax().item()
    best_box = boxes[best_idx]

    # 将坐标转换回原始图像尺寸
    best_box = [
        best_box[0] / scale,
        best_box[1] / scale,
        best_box[2] / scale,
        best_box[3] / scale,
    ]

    return best_box, scores.cpu().numpy(), boxes


def visualize_result(image_path, best_box, output_path):
    """可视化结果"""
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # 绘制最佳裁剪框
    x1, y1, x2, y2 = map(int, best_box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 保存结果
    cv2.imwrite(output_path, image)
    print(f"结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='图像裁剪推理')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    parser.add_argument('--caption', type=str, default='a photo', help='图像情感描述')
    parser.add_argument('--num_candidates', type=int, default=64, help='候选框数量')
    parser.add_argument('--output', type=str, default='output.jpg', help='输出图像路径')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model = load_model(args.model, config, device)
    print("模型加载完成")

    # 推理
    best_box, scores, boxes = inference(
        model, args.image, args.caption, args.num_candidates, device
    )

    print(f"\n最佳裁剪框: {best_box}")
    print(f"分数: {scores.max():.4f}")

    # 可视化
    visualize_result(args.image, best_box, args.output)


if __name__ == '__main__':
    main()
