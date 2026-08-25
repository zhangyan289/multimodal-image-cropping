"""
损失函数模块
提供多种损失函数组合用于训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    """回归损失（Smooth L1 / MSE / Weighted L1）"""

    def __init__(self, loss_type="smooth_l1", beta=1.0, weighted=False):
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta
        self.weighted = weighted

    def forward(self, pred_scores, target_scores):
        """
        Args:
            pred_scores: 预测分数 [N]
            target_scores: 真实分数 [N]
        Returns:
            loss: 标量
        """
        if self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(pred_scores, target_scores, reduction='none', beta=self.beta)
        elif self.loss_type == "mse":
            loss = F.mse_loss(pred_scores, target_scores, reduction='none')
        elif self.loss_type == "weighted_l1":
            loss = F.l1_loss(pred_scores, target_scores, reduction='none')
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")

        if self.weighted:
            # 根据分数大小加权（高分样本更重要）
            weights = torch.exp((target_scores - target_scores.mean()).clamp(min=0, max=10))
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()

        return loss


class RankingLoss(nn.Module):
    """
    排序损失
    确保好的裁剪比差的裁剪获得更高的分数
    """

    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, pred_scores, target_scores):
        """
        Args:
            pred_scores: 预测分数 [N]
            target_scores: 真实分数 [N]
        Returns:
            loss: 标量
        """
        N = pred_scores.shape[0]
        if N <= 1:
            return torch.tensor(0.0, device=pred_scores.device)

        # 计算所有样本对的分数差
        pred_diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)  # [N, N]
        target_diff = target_scores.unsqueeze(1) - target_scores.unsqueeze(0)  # [N, N]

        # 只考虑真实分数有差异的样本对
        mask = target_diff.abs() > self.margin

        # 如果真实分数 i > j，则预测分数也应该 i > j
        # 即 target_diff > 0 时，pred_diff 也应该 > 0
        indicator = -torch.sign(target_diff) * (pred_diff - target_diff)
        indicator = torch.maximum(indicator, torch.zeros_like(indicator))

        # 只计算有效样本对
        if mask.sum() > 0:
            loss = (indicator * mask.float()).sum() / mask.sum()
        else:
            loss = torch.tensor(0.0, device=pred_scores.device)

        return loss


class ContrastiveLoss(nn.Module):
    """
    对比损失
    拉大好坏裁剪之间的距离
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, target_scores):
        """
        Args:
            features: 融合后的特征 [N, D]
            target_scores: 真实分数 [N]
        Returns:
            loss: 标量
        """
        N = features.shape[0]
        if N <= 1:
            return torch.tensor(0.0, device=features.device)

        # 归一化特征
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵
        similarity = torch.matmul(features, features.T) / self.temperature

        # 根据分数差异构建标签
        # 分数相近的样本应该相似，分数差异大的应该不相似
        score_diff = (target_scores.unsqueeze(1) - target_scores.unsqueeze(0)).abs()
        labels = (score_diff < 0.5).float()  # 分数差 < 0.5 的视为正样本对

        # 排除对角线
        labels = labels - torch.eye(N, device=features.device)

        # 计算对比损失（简化版）
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # 只计算正样本对的损失
        if labels.sum() > 0:
            loss = -(log_prob * labels).sum() / labels.sum()
        else:
            loss = torch.tensor(0.0, device=features.device)

        return loss


class CroppingLoss(nn.Module):
    """
    组合损失函数
    整合回归损失、排序损失和对比损失
    """

    def __init__(
        self,
        regression_type="smooth_l1",
        regression_weight=1.0,
        ranking_weight=0.5,
        contrastive_weight=0.1,
        smooth_l1_beta=1.0,
    ):
        super().__init__()
        self.regression_loss = RegressionLoss(
            loss_type=regression_type,
            beta=smooth_l1_beta,
            weighted=(regression_type == "weighted_l1")
        )
        self.ranking_loss = RankingLoss()
        self.contrastive_loss = ContrastiveLoss()

        self.regression_weight = regression_weight
        self.ranking_weight = ranking_weight
        self.contrastive_weight = contrastive_weight

    def forward(self, pred_scores, target_scores, features=None):
        """
        Args:
            pred_scores: 预测分数 [N]
            target_scores: 真实分数 [N]
            features: 融合特征 [N, D]（可选，用于对比损失）
        Returns:
            loss_dict: 包含各项损失和总损失的字典
        """
        # 回归损失
        reg_loss = self.regression_loss(pred_scores, target_scores)

        # 排序损失
        rank_loss = self.ranking_loss(pred_scores, target_scores)

        # 对比损失（如果有特征）
        if features is not None and self.contrastive_weight > 0:
            cont_loss = self.contrastive_loss(features, target_scores)
        else:
            cont_loss = torch.tensor(0.0, device=pred_scores.device)

        # 总损失
        total_loss = (
            self.regression_weight * reg_loss +
            self.ranking_weight * rank_loss +
            self.contrastive_weight * cont_loss
        )

        return {
            'total_loss': total_loss,
            'regression_loss': reg_loss,
            'ranking_loss': rank_loss,
            'contrastive_loss': cont_loss,
        }
