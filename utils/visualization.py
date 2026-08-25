"""
可视化工具模块
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_cropping_results(image_path, boxes, scores, output_path=None, top_k=5):
    """
    可视化裁剪结果

    Args:
        image_path: 原始图像路径
        boxes: 候选框列表 [[x1, y1, x2, y2], ...]
        scores: 每个候选框的分数
        output_path: 输出图像路径（可选）
        top_k: 显示 top-k 个最佳裁剪
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 按分数排序
    sorted_indices = np.argsort(scores)[::-1]

    fig, axes = plt.subplots(1, top_k + 1, figsize=(20, 4))

    # 显示原始图像和所有候选框
    axes[0].imshow(image_rgb)
    for i, idx in enumerate(sorted_indices[:20]):  # 显示前20个
        box = boxes[idx]
        alpha = 0.3 + 0.7 * (scores[idx] - scores.min()) / (scores.max() - scores.min() + 1e-6)
        color = plt.cm.RdYlGn(alpha)
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor=color,
            facecolor='none'
        )
        axes[0].add_patch(rect)
    axes[0].set_title(f"All Candidates (showing top 20)")
    axes[0].axis('off')

    # 显示 top-k 最佳裁剪
    for k in range(top_k):
        idx = sorted_indices[k]
        box = boxes[idx]
        score = scores[idx]

        # 裁剪图像
        x1, y1, x2, y2 = map(int, box)
        cropped = image_rgb[y1:y2, x1:x2]

        axes[k + 1].imshow(cropped)
        axes[k + 1].set_title(f"Top {k + 1}\nScore: {score:.3f}")
        axes[k + 1].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_training_curves(log_file, output_path=None):
    """
    绘制训练曲线

    Args:
        log_file: 训练日志文件路径
        output_path: 输出图像路径（可选）
    """
    # 这里需要根据实际的日志格式进行解析
    # 简化版本，实际需要适配具体的日志格式
    print("训练曲线绘制功能需要根据实际日志格式实现")
