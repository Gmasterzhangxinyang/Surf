#!/usr/bin/env python3
"""
BEV Comparison Visualization - BEV对比可视化

显示: 6张相机图像 + Baseline BEV + Agent BEV + GT + IoU对比
"""

import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BEVConfig
from data.nuscenes_loader import NuScenesLoader
from data.bev_gt import generate_bev_gt
from agent.core import AgentCore
from agent.bev_evaluator import BEVEvaluator
from models.bevfusion import BEVFusion


CAMERA_NAMES = {
    0: "CAM_FRONT",
    1: "CAM_FRONT_RIGHT",
    2: "CAM_FRONT_LEFT",
    3: "CAM_BACK",
    4: "CAM_BACK_RIGHT",
    5: "CAM_BACK_LEFT"
}


def tensor_to_image(img_tensor):
    """将tensor转为numpy图像 (H, W, 3)"""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]  # 取batch第一个
    if img_tensor.dim() == 3:
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        img = img_tensor.cpu().numpy()

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img


def create_bev_visualization(bev_np):
    """创建BEV可视化 (黑底白点)"""
    h, w = bev_np.shape
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    viz[bev_np == 1] = [255, 255, 255]  # 占用=白色
    return viz


def run_comparison(args):
    """运行对比实验"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = BEVConfig(device=device)

    print("=" * 60)
    print("BEV Comparison Visualization")
    print("=" * 60)
    print(f"Device: {device}")

    # 加载数据
    print("\nLoading nuScenes...")
    loader = NuScenesLoader(args.dataroot, args.version, cfg)
    sample = loader[args.sample]
    raw_sample = loader.samples[args.sample]

    # 准备输入
    images = sample["images"].to(device)  # (B, 6, 3, H, W)
    intrinsics = sample["intrinsics"].to(device)
    extrinsics = sample["extrinsics"].to(device)
    lidar_points = sample["lidar_points"].to(device)
    lidar_mask = sample["lidar_mask"].to(device)

    # 生成GT BEV
    gt_bev = generate_bev_gt(loader.nusc, raw_sample, cfg).cpu().numpy()

    # 加载模型
    print("\nBuilding model...")
    model = BEVFusion(cfg).to(device)
    model.eval()

    model_path = "best_model.pth"  # 用户提供的新权重
    if os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

    # BEV配置
    bev_cfg = {
        "bev_x_range": cfg.bev_x_range,
        "bev_y_range": cfg.bev_y_range,
        "bev_size": cfg.bev_size,
        "image_size": cfg.image_size,
    }

    # 初始化
    agent = AgentCore(llm_url=args.llm_url, max_iterations=args.max_iterations, fast_mode=getattr(args, 'fast', False))
    evaluator = BEVEvaluator()

    # Baseline: 原始BEVFusion
    print("\nGenerating Baseline BEV...")
    with torch.no_grad():
        logits_baseline, bev_baseline = model(images, intrinsics, extrinsics, lidar_points, lidar_mask)
        bev_baseline = bev_baseline[0] if bev_baseline.dim() > 2 else bev_baseline
        bev_baseline_np = bev_baseline.cpu().numpy()

    # 评估Baseline
    gt_tensor = torch.from_numpy(gt_bev).long().to(device)
    baseline_eval = evaluator.evaluate_with_gt(bev_baseline, gt_tensor)

    # Agent优化后
    print("Running Agent optimization...")
    with torch.no_grad():
        agent_result = agent.run(model, images, intrinsics, extrinsics, lidar_points, lidar_mask, bev_cfg)

    bev_agent = agent_result["final_bev"]
    if bev_agent.dim() > 2:
        bev_agent = bev_agent[0]
    bev_agent_np = bev_agent.cpu().numpy()

    # 评估Agent结果
    agent_eval = evaluator.evaluate_with_gt(bev_agent, gt_tensor)

    # 打印结果
    print(f"\n结果:")
    print(f"  Baseline IoU: {baseline_eval['iou']:.3f}, Accuracy: {baseline_eval['accuracy']:.3f}")
    print(f"  Agent IoU: {agent_eval['iou']:.3f}, Accuracy: {agent_eval['accuracy']:.3f}")
    print(f"  IoU提升: {agent_eval['iou'] - baseline_eval['iou']:+.3f}")

    # ==================== 创建可视化 ====================
    print("\n生成对比图像...")

    # 获取6张相机图像
    cam_images = []
    for i in range(6):
        img = tensor_to_image(images[0, i])
        cam_images.append(img)

    # 创建BEV可视化
    bev_baseline_viz = create_bev_visualization(bev_baseline_np)
    bev_agent_viz = create_bev_visualization(bev_agent_np)
    gt_viz = create_bev_visualization(gt_bev)

    # 创建差异图
    diff = (bev_agent_np.astype(int) - bev_baseline_np.astype(int))
    diff_viz = np.zeros((*diff.shape, 3), dtype=np.uint8)
    diff_viz[diff == 1] = [0, 200, 0]      # FP = 绿色 (优化后多了)
    diff_viz[diff == -1] = [200, 0, 0]    # FN = 红色 (优化后少了)
    # 正确=灰色
    correct = (bev_agent_np == bev_baseline_np)
    diff_viz[correct] = [80, 80, 80]

    # 创建大图布局
    # 行1: 6张相机图像
    # 行2: Baseline BEV | Agent BEV | GT | 差异图
    # 行3: 决策历程
    fig = plt.figure(figsize=(24, 18))

    # 第一行: 6张相机图像 (标注问题区域)
    problem_cams = set()
    for h in agent_result['history']:
        if 'decision' in h:
            action = h['decision'].get('action', {})
            params = action.get('parameters', {})
            for cid in params.get('camera_ids', []):
                problem_cams.add(cid)

    for i, (cam_img, cam_name) in enumerate(zip(cam_images, CAMERA_NAMES.values())):
        ax = fig.add_subplot(2, 6, i + 1)
        cam_cv = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR).copy()

        # 如果这个相机被处理过，标注
        if i in problem_cams:
            # 找到对应的action
            for h in agent_result['history']:
                if 'decision' in h:
                    action = h['decision'].get('action', {})
                    params = action.get('parameters', {})
                    if i in params.get('camera_ids', []):
                        tool = action.get('name', 'unknown')
                        thought = h['decision'].get('thought', '')[:30]
                        # 画红框
                        cv2.rectangle(cam_cv, (5, 5), (cam_img.shape[1]-5, cam_img.shape[0]-5),
                                     (0, 0, 255), 2)
                        # 标注工具名
                        cv2.putText(cam_cv, f'{tool}', (10, 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        break

        ax.imshow(cam_cv)
        ax.set_title(f'{cam_name}', fontsize=11)
        ax.axis('off')

    # 第二行: BEV对比
    # Baseline BEV
    ax = fig.add_subplot(2, 6, 7)
    ax.imshow(bev_baseline_viz)
    ax.set_title(f'Baseline BEV\nIoU: {baseline_eval["iou"]:.3f}', fontsize=12, color='blue')
    ax.axis('off')

    # Agent BEV
    ax = fig.add_subplot(2, 6, 8)
    ax.imshow(bev_agent_viz)
    iou_change = agent_eval['iou'] - baseline_eval['iou']
    color = 'green' if iou_change > 0 else 'red' if iou_change < 0 else 'black'
    ax.set_title(f'Agent BEV\nIoU: {agent_eval["iou"]:.3f} ({iou_change:+.3f})', fontsize=12, color=color)
    ax.axis('off')

    # GT
    ax = fig.add_subplot(2, 6, 9)
    ax.imshow(gt_viz)
    ax.set_title('Ground Truth', fontsize=12)
    ax.axis('off')

    # 差异图
    ax = fig.add_subplot(2, 6, 10)
    ax.imshow(diff_viz)
    ax.set_title('Difference Map\nGreen=FP, Red=FN, Gray=Same', fontsize=11)
    ax.axis('off')

    # 留空
    ax = fig.add_subplot(3, 6, 11)
    ax.axis('off')

    # 图例
    ax = fig.add_subplot(3, 6, 12)
    ax.axis('off')
    green_patch = mpatches.Patch(color=[0, 0.78, 0], label='False Positive (Agent预测1但实际0)')
    red_patch = mpatches.Patch(color=[0.78, 0, 0], label='False Negative (Agent预测0但实际1)')
    gray_patch = mpatches.Patch(color=[0.31, 0.31, 0.31], label='与Baseline相同')
    redbox_patch = mpatches.Patch(color=[1, 0, 0], label='红框=Agent处理过的相机')
    ax.legend(handles=[green_patch, red_patch, gray_patch, redbox_patch], loc='center', fontsize=9)

    # 第3行: 决策历程面板
    ax = fig.add_subplot(3, 1, 3)
    ax.axis('off')

    # 构建决策历史文本
    history_text = "=" * 60 + "\n"
    history_text += "Agent 决策历程 (Agent Decision History)\n"
    history_text += "=" * 60 + "\n\n"

    for h in agent_result['history']:
        if 'decision' in h:
            iteration = h.get('iteration', 0)
            decision = h['decision']
            thought = decision.get('thought', 'N/A')
            action = decision.get('action', {})
            action_name = action.get('name', 'N/A')
            params = action.get('parameters', {})

            history_text += f"[Iteration {iteration}]\n"
            history_text += f"  Thought: {thought}\n"
            history_text += f"  Action:  {action_name}\n"
            history_text += f"  Params:  camera_ids={params.get('camera_ids', [])}\n"
            history_text += f"           enhancement_type={params.get('enhancement_type', 'N/A')}\n"
            history_text += "-" * 40 + "\n"

    history_text += f"\n结果: Baseline IoU={baseline_eval['iou']:.3f} → Agent IoU={agent_eval['iou']:.3f} ({agent_eval['iou']-baseline_eval['iou']:+.3f})\n"

    ax.text(0.02, 0.95, history_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 总标题
    improvement = agent_eval['iou'] - baseline_eval['iou']
    fig.suptitle(f'Sample {args.sample} - BEV Comparison | IoU Improvement: {improvement:+.3f}',
                 fontsize=16, y=0.98)

    plt.tight_layout()

    # 保存
    output_path = args.output or f"bev_comparison_sample{args.sample}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n对比图像已保存: {output_path}")

    # 打印决策历史
    print("\nAgent决策历史:")
    for h in agent_result['history']:
        if 'decision' in h:
            print(f"  Iteration {h['iteration']}: {h['decision'].get('thought', 'N/A')}")
            action = h['decision'].get('action', {})
            print(f"    Action: {action.get('name', 'N/A')}, params: {action.get('parameters', {})}")

    # 保存单独的BEV图像
    plt.figure(figsize=(8, 8))
    plt.imshow(bev_baseline_viz)
    plt.title(f'Baseline BEV (IoU={baseline_eval["iou"]:.3f})', fontsize=14)
    plt.axis('off')
    plt.savefig(f"bev_baseline_sample{args.sample}.png", dpi=150, bbox_inches='tight')
    print(f"原始BEV已保存: bev_baseline_sample{args.sample}.png")

    plt.figure(figsize=(8, 8))
    plt.imshow(bev_agent_viz)
    plt.title(f'Agent BEV (IoU={agent_eval["iou"]:.3f})', fontsize=14)
    plt.axis('off')
    plt.savefig(f"bev_agent_sample{args.sample}.png", dpi=150, bbox_inches='tight')
    print(f"优化后BEV已保存: bev_agent_sample{args.sample}.png")

    return baseline_eval, agent_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEV Comparison Visualization")
    parser.add_argument("--dataroot", type=str, default="./v1.0-mini", help="Path to nuScenes")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="Dataset version")
    parser.add_argument("--sample", type=int, default=None, help="Single sample index (if not set, run batch)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test in batch mode")
    parser.add_argument("--max_iterations", type=int, default=3, help="Max agent iterations")
    parser.add_argument("--llm_url", type=str, default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--fast", action="store_true", help="Fast mode: skip VisionLLM, use rule-based decisions")

    args = parser.parse_args()

    # 批量测试模式
    if args.sample is None:
        print(f"=" * 60)
        print(f"批量测试模式: 测试 {args.num_samples} 个样本")
        print(f"=" * 60)

        from data.nuscenes_loader import NuScenesLoader
        from config import BEVConfig
        cfg = BEVConfig(device="cpu")
        loader = NuScenesLoader(args.dataroot, args.version, cfg)
        total = min(args.num_samples, len(loader))

        all_results = []
        improved_count = 0
        degraded_count = 0

        for idx in range(total):
            print(f"\n处理样本 {idx}/{total}...")
            args.sample = idx
            args.output = f"bev_comparison_sample{idx}.png"
            try:
                baseline_eval, agent_eval = run_comparison(args)
                iou_diff = agent_eval['iou'] - baseline_eval['iou']
                all_results.append({
                    'idx': idx,
                    'baseline_iou': baseline_eval['iou'],
                    'agent_iou': agent_eval['iou'],
                    'iou_diff': iou_diff,
                    'improved': iou_diff > 0
                })
                if iou_diff > 0:
                    improved_count += 1
                elif iou_diff < 0:
                    degraded_count += 1
            except Exception as e:
                print(f"样本 {idx} 处理失败: {e}")

        # 打印汇总
        print(f"\n{'=' * 60}")
        print("批量测试汇总")
        print(f"{'=' * 60}")
        avg_iou = sum(r['iou_diff'] for r in all_results) / len(all_results) if all_results else 0
        print(f"测试样本数: {len(all_results)}")
        print(f"IoU提升: 平均 {avg_iou:+.4f}")
        print(f"提升样本: {improved_count}/{len(all_results)}")
        print(f"下降样本: {degraded_count}/{len(all_results)}")

        # 保存汇总图
        import matplotlib.pyplot as plt
        if all_results:
            fig, ax = plt.subplots(figsize=(12, 6))
            idxs = [r['idx'] for r in all_results]
            baseline_ious = [r['baseline_iou'] for r in all_results]
            agent_ious = [r['agent_iou'] for r in all_results]
            colors = ['green' if r['iou_diff'] > 0 else 'red' for r in all_results]

            x = range(len(idxs))
            ax.bar([i-0.2 for i in x], baseline_ious, 0.4, label='Baseline', color='blue', alpha=0.6)
            ax.bar([i+0.2 for i in x], agent_ious, 0.4, label='Agent', color='orange', alpha=0.6)

            ax.set_xlabel('Sample Index')
            ax.set_ylabel('IoU')
            ax.set_title(f'Batch Results: {len(all_results)} samples | Avg IoU Change: {avg_iou:+.4f}')
            ax.legend()
            ax.set_xticks(x)
            ax.set_xticklabels(idxs)

            plt.tight_layout()
            plt.savefig('batch_comparison_summary.png', dpi=150)
            print(f"\n汇总图已保存: batch_comparison_summary.png")
    else:
        # 单样本模式
        run_comparison(args)