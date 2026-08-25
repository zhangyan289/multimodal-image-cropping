"""
数据集加载模块
支持 GAICD 等图像裁剪数据集

数据输入：
- 图像图：原始输入图像
- 文字图：大模型生成的描述，包含内容描述和情感描述两部分

训练策略：将情感单独作为特征进行训练
"""
import os
import cv2
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CroppingDataset(Dataset):
    """
    图像裁剪数据集

    数据目录结构:
        dataset_dir/
        ├── images/
        │   ├── train/
        │   │   ├── 001.jpg
        │   │   └── ...
        │   └── test/
        ├── annotations/
        │   ├── train/
        │   │   ├── 001.txt  # 每行: y1 x1 y2 x2 score
        │   │   └── ...
        │   └── test/
        └── captions/
            ├── train.csv    # image_id,semantic_caption,emotion_caption
            └── test.csv

    提示词升级：文字图包含内容描述（semantic_caption）和情感描述（emotion_caption）
    训练策略：将情感单独作为特征进行训练

    Args:
        dataset_dir: 数据集根目录
        split: 数据集划分 (train/test)
        image_size: 图像缩放尺寸
        rgb_mean: RGB 均值
        rgb_std: RGB 标准差
        augmentation: 是否使用数据增强
    """

    MOS_MEAN = 2.95
    MOS_STD = 0.8

    def __init__(
        self,
        dataset_dir="./data/GAIC",
        split="train",
        image_size=224,
        rgb_mean=(0.48145466, 0.4578275, 0.40821073),
        rgb_std=(0.26862954, 0.26130258, 0.27577711),
        augmentation=False,
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.image_size = image_size
        self.rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(rgb_std, dtype=np.float32)
        self.augmentation = augmentation

        # 加载图像列表
        image_dir = os.path.join(dataset_dir, "images", split)
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")

        self.image_list = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        # 加载注释
        self.annotations = {}
        ann_dir = os.path.join(dataset_dir, "annotations", split)
        for img_name in self.image_list:
            img_id = os.path.splitext(img_name)[0]
            ann_file = os.path.join(ann_dir, f"{img_id}.txt")
            if os.path.exists(ann_file):
                self.annotations[img_id] = self._load_annotations(ann_file)
            else:
                self.annotations[img_id] = []

        # 加载文字图描述（由千问大模型通过情感感知提示词生成）
        # 包含两部分：semantic_caption（内容描述）和 emotion_caption（情感描述）
        self.semantic_captions = {}  # 内容描述
        self.emotion_captions = {}   # 情感描述（独立特征）
        
        caption_file = os.path.join(dataset_dir, "captions", f"{split}.csv")
        if os.path.exists(caption_file):
            df = pd.read_csv(caption_file)
            # 支持新的三列格式：image_id, semantic_caption, emotion_caption
            if 'semantic_caption' in df.columns and 'emotion_caption' in df.columns:
                self.semantic_captions = pd.Series(df['semantic_caption'].values, index=df['image_id']).to_dict()
                self.emotion_captions = pd.Series(df['emotion_caption'].values, index=df['image_id']).to_dict()
            # 兼容旧的两列格式：image_id, caption（将 caption 同时作为 semantic 和 emotion）
            elif 'caption' in df.columns:
                captions = pd.Series(df['caption'].values, index=df['image_id']).to_dict()
                self.semantic_captions = captions
                self.emotion_captions = captions
            else:
                raise ValueError(f"CSV 文件格式错误，需要包含 image_id, semantic_caption, emotion_caption 列")
        else:
            # 如果没有文字图，使用默认描述
            for img_name in self.image_list:
                img_id = os.path.splitext(img_name)[0]
                self.semantic_captions[img_id] = "a photo"
                self.emotion_captions[img_id] = "a peaceful and harmonious scene"

    def _load_annotations(self, ann_file):
        """加载注释文件"""
        annotations = []
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    y1, x1, y2, x2, score = map(float, parts[:5])
                    if score != -2:  # 过滤无效标注
                        annotations.append({
                            'bbox': [x1, y1, x2, y2],
                            'score': score
                        })
        return annotations

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.dataset_dir, "images", self.split, img_name)

        # 加载图像图
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取注释
        annotations = self.annotations.get(img_id, [])

        # 获取文字图描述（包含内容描述和情感描述）
        semantic_caption = self.semantic_captions.get(img_id, "a photo")
        emotion_caption = self.emotion_captions.get(img_id, "a peaceful and harmonious scene")

        # 数据预处理
        image, annotations = self._preprocess(image, annotations)

        # 准备输出
        bboxes = []
        scores = []
        for ann in annotations:
            bboxes.append(ann['bbox'])
            scores.append(ann['score'])

        if len(bboxes) == 0:
            # 如果没有有效注释，返回一个默认框
            bboxes = [[0, 0, image.shape[2], image.shape[1]]]
            scores = [0.0]

        return {
            'image': image,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'scores': torch.tensor(scores, dtype=torch.float32),
            'semantic_caption': semantic_caption,  # 内容描述
            'emotion_caption': emotion_caption,    # 情感描述（独立特征）
            'image_id': img_id,
        }

    def _preprocess(self, image, annotations):
        """图像预处理"""
        h, w = image.shape[:2]

        # 缩放图像
        scale = self.image_size / min(h, w)
        new_h = int(round(h * scale / 32) * 32)
        new_w = int(round(w * scale / 32) * 32)
        image = cv2.resize(image, (new_w, new_h))

        # 归一化
        image = image.astype(np.float32) / 255.0
        image = (image - self.rgb_mean) / self.rgb_std

        # 转换为 CHW 格式
        image = image.transpose(2, 0, 1)

        # 缩放注释
        scale_h = new_h / h
        scale_w = new_w / w
        for ann in annotations:
            bbox = ann['bbox']
            ann['bbox'] = [
                bbox[0] * scale_w,  # x1
                bbox[1] * scale_h,  # y1
                bbox[2] * scale_w,  # x2
                bbox[3] * scale_h,  # y2
            ]
            # 标准化分数
            ann['score'] = (ann['score'] - self.MOS_MEAN) / self.MOS_STD

        # 数据增强（可选）
        if self.augmentation:
            image, annotations = self._augment(image, annotations)

        return image, annotations

    def _augment(self, image, annotations):
        """简单的数据增强"""
        # 随机水平翻转
        if np.random.rand() > 0.5:
            image = image[:, :, ::-1].copy()
            w = image.shape[2]
            for ann in annotations:
                bbox = ann['bbox']
                ann['bbox'] = [
                    w - bbox[2],  # x1
                    bbox[1],       # y1
                    w - bbox[0],  # x2
                    bbox[3],       # y2
                ]

        return image, annotations
