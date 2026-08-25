"""
评估指标模块
提供 SRCC、PCC、Acc@K 等评估指标
"""
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
import math


class CroppingMetrics:
    """
    图像裁剪评估指标集合
    """

    def __init__(self, top_k_list=(4, 8)):
        self.top_k_list = top_k_list
        self.reset()

    def reset(self):
        """重置所有指标"""
        self.all_preds = []
        self.all_targets = []
        self.acc_at_k = {k: [] for k in self.top_k_list}
        self.wacc_at_k = {k: [] for k in self.top_k_list}

    def update(self, pred_scores, target_scores, bboxes=None):
        """
        更新指标（单张图片的所有候选框）

        Args:
            pred_scores: 预测分数 [N]
            target_scores: 真实分数 [N]
            bboxes: 候选框坐标 [N, 4]（可选）
        """
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().detach().numpy()
        if isinstance(target_scores, torch.Tensor):
            target_scores = target_scores.cpu().detach().numpy()

        self.all_preds.extend(pred_scores.tolist())
        self.all_targets.extend(target_scores.tolist())

        # 计算 Top-K 准确率
        N = len(pred_scores)
        if N == 0:
            return

        # 按真实分数排序
        target_sorted_indices = np.argsort(target_scores)[::-1]
        pred_sorted_indices = np.argsort(pred_scores)[::-1]

        for k in self.top_k_list:
            # Acc@K: Top-K 预测中包含最佳裁剪的比例
            top_k_pred_indices = pred_sorted_indices[:k]
            best_k_target = target_scores[target_sorted_indices[min(4, N-1)]]

            acc = 0.0
            for idx in top_k_pred_indices:
                if target_scores[idx] >= best_k_target:
                    acc += 1.0
            self.acc_at_k[k].append(acc / k)

            # Weighted Acc@K: 考虑排名位置的加权准确率
            wacc = 0.0
            rank_of_returned = [target_sorted_indices.tolist().index(idx) for idx in top_k_pred_indices]
            rank_of_returned.sort()

            for j, rank in enumerate(rank_of_returned):
                if rank <= min(4, N-1):
                    wacc += 1.0 * math.exp(-0.2 * (rank - j))
            self.wacc_at_k[k].append(wacc / k)

    def compute(self):
        """
        计算所有指标

        Returns:
            metrics_dict: 包含所有指标的字典
        """
        if len(self.all_preds) == 0:
            return {}

        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)

        # 计算 SRCC 和 PCC（按图片分组计算，然后平均）
        # 这里简化为全局计算
        srcc, _ = spearmanr(preds, targets)
        pcc, _ = pearsonr(preds, targets)

        # 计算平均 Top-K 准确率
        acc_at_k = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in self.acc_at_k.items()}
        wacc_at_k = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in self.wacc_at_k.items()}

        return {
            'srcc': srcc,
            'pcc': pcc,
            'acc_at_4': acc_at_k.get(4, 0.0),
            'acc_at_8': acc_at_k.get(8, 0.0),
            'wacc_at_4': wacc_at_k.get(4, 0.0),
            'wacc_at_8': wacc_at_k.get(8, 0.0),
        }

    def __str__(self):
        metrics = self.compute()
        if not metrics:
            return "No metrics computed"

        return (
            f"SRCC: {metrics['srcc']:.4f} | "
            f"PCC: {metrics['pcc']:.4f} | "
            f"Acc@4: {metrics['acc_at_4']:.4f} | "
            f"Acc@8: {metrics['acc_at_8']:.4f} | "
            f"wAcc@4: {metrics['wacc_at_4']:.4f}"
        )
