#!/usr/bin/env python3
"""
Agent Inference Entry - Agent推理入口

对比实验设计：
同一输入 → 原始BEVFusion → BEV_A → 与GT计算IoU_A
同一输入 → BEVFusion + Agent → BEV_B → 与GT计算IoU_B
                                 ↓
                         IoU_B - IoU_A = 提升效果
"""

import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BEVConfig
from models.bevfusion import BEVFusion
from data.nuscenes_loader import NuScenesLoader
from data.bev_gt import generate_bev_gt
from agent.core import AgentCore
from agent.bev_evaluator import BEVEvaluator
from agent.data_logger import DataLogger


def run_agent_inference(args):
    """运行Agent推理"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = BEVConfig(device=device)

    print("=" * 60)
    print("BEVFusion + Active Agent Inference")
    print("=" * 60)
    print(f"Device: {device}")

    # 加载数据
    print("\nLoading nuScenes...")
    loader = NuScenesLoader(args.dataroot, args.version, cfg)
    print(f"Total samples: {len(loader)}")

    # 加载模型
    print("\nBuilding model...")
    model = BEVFusion(cfg)
    model = model.to(device)
    model.eval()

    # 加载训练好的权重
    model_path = "train_output/best_model.pth"
    if os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Warning: No trained weights found, using random initialization")

    # BEV配置 - 传给Agent用于几何映射
    bev_cfg = {
        "bev_x_range": cfg.bev_x_range,
        "bev_y_range": cfg.bev_y_range,
        "bev_size": cfg.bev_size,
        "image_size": cfg.image_size,  # (H, W) tuple
    }

    # 初始化
    agent = AgentCore(
        llm_url=args.llm_url,
        max_iterations=args.max_iterations
    )
    evaluator = BEVEvaluator()
    data_logger = DataLogger(args.log_file)

    # 选择样本
    if args.sample is not None:
        sample_indices = [args.sample]
    else:
        sample_indices = list(range(0, min(args.num_samples, len(loader))))

    results = []

    for idx in sample_indices:
        print(f"\n{'='*60}")
        print(f"Processing sample {idx}...")
        print("=" * 60)

        sample = loader[idx]
        raw_sample = loader.samples[idx]  # 原始nuScenes sample dict

        # 准备输入
        images = sample["images"].to(device)
        intrinsics = sample["intrinsics"].to(device)
        extrinsics = sample["extrinsics"].to(device)
        lidar_points = sample["lidar_points"].to(device)
        lidar_mask = sample["lidar_mask"].to(device)

        # 生成GT BEV (需要原始sample dict)
        gt_bev = generate_bev_gt(loader.nusc, raw_sample, cfg)
        gt_bev_tensor = gt_bev.long().to(device)

        # Baseline: 原始BEVFusion
        with torch.no_grad():
            logits_baseline, bev_baseline = model(images, intrinsics, extrinsics, lidar_points, lidar_mask)
            bev_baseline = bev_baseline[0] if bev_baseline.dim() > 2 else bev_baseline

        # 评估Baseline
        baseline_eval = evaluator.evaluate_with_gt(bev_baseline, gt_bev_tensor)
        print(f"\nBaseline (无Agent):")
        print(f"  IoU: {baseline_eval['iou']:.3f}")
        print(f"  Accuracy: {baseline_eval['accuracy']:.3f}")

        # Agent优化后
        with torch.no_grad():
            agent_result = agent.run(
                model, images, intrinsics, extrinsics,
                lidar_points, lidar_mask, bev_cfg
            )

        bev_agent = agent_result["final_bev"]
        if bev_agent.dim() > 2:
            bev_agent = bev_agent[0]

        # 评估Agent结果
        agent_eval = evaluator.evaluate_with_gt(bev_agent, gt_bev_tensor)
        print(f"\nAgent优化后:")
        print(f"  IoU: {agent_eval['iou']:.3f}")
        print(f"  Accuracy: {agent_eval['accuracy']:.3f}")
        print(f"  Iterations: {len([h for h in agent_result['history'] if 'decision' in h])}")

        # 计算提升
        iou_improvement = agent_eval['iou'] - baseline_eval['iou']
        acc_improvement = agent_eval['accuracy'] - baseline_eval['accuracy']

        print(f"\n提升效果:")
        print(f"  IoU: {baseline_eval['iou']:.3f} → {agent_eval['iou']:.3f} ({iou_improvement:+.3f})")
        print(f"  Accuracy: {baseline_eval['accuracy']:.3f} → {agent_eval['accuracy']:.3f} ({acc_improvement:+.3f})")

        # 打印决策历史（含相机映射信息）
        for h in agent_result['history']:
            if 'decision' in h:
                print(f"\n  Iteration {h['iteration']}:")
                print(f"    Thought: {h['decision'].get('thought', 'N/A')}")
                action = h['decision'].get('action', {})
                print(f"    Action: {action.get('name', 'N/A')}")
                print(f"    Camera IDs: {action.get('parameters', {}).get('camera_ids', 'all')}")

                # 记录数据
                data_logger.log(
                    iteration=h['iteration'],
                    input_state={"sample_idx": idx},
                    bev_quality={
                        "iou": agent_eval['iou'],
                        "accuracy": agent_eval['accuracy']
                    },
                    agent_output=h['decision'],
                    result={
                        "iou_improvement": iou_improvement,
                        "improved": iou_improvement > 0
                    }
                )

        results.append({
            "sample_idx": idx,
            "baseline_iou": baseline_eval['iou'],
            "agent_iou": agent_eval['iou'],
            "iou_improvement": iou_improvement,
            "baseline_acc": baseline_eval['accuracy'],
            "agent_acc": agent_eval['accuracy'],
            "acc_improvement": acc_improvement,
            "iterations": len([h for h in agent_result['history'] if 'decision' in h])
        })

    # 汇总统计
    print(f"\n{'='*60}")
    print("汇总统计")
    print("=" * 60)

    total_iou_improvement = sum(r['iou_improvement'] for r in results)
    avg_iou_improvement = total_iou_improvement / len(results) if results else 0

    improved_count = sum(1 for r in results if r['iou_improvement'] > 0)
    degraded_count = sum(1 for r in results if r['iou_improvement'] < 0)

    print(f"样本数: {len(results)}")
    print(f"IoU提升: 平均{avg_iou_improvement:+.3f}")
    print(f"提升样本: {improved_count}/{len(results)}")
    print(f"下降样本: {degraded_count}/{len(results)}")

    # 数据分析
    print(f"\n{'='*60}")
    print("数据分析")
    print("=" * 60)
    stats = data_logger.analyze()
    if stats:
        for action_name, stat in stats.items():
            print(f"  {action_name}: {stat['count']}次, 成功率{stat.get('success_rate', 0):.1%}")
    else:
        print("  暂无数据记录")


def main():
    parser = argparse.ArgumentParser(description="BEVFusion + Active Agent Inference")
    parser.add_argument("--dataroot", type=str, default="./v1.0-mini", help="Path to nuScenes")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="Dataset version")
    parser.add_argument("--sample", type=int, default=None, help="Sample index")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--max_iterations", type=int, default=3, help="Max agent iterations")
    parser.add_argument("--llm_url", type=str, default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--log_file", type=str, default="agent_training_data.jsonl", help="Log file path")
    args = parser.parse_args()

    run_agent_inference(args)


if __name__ == "__main__":
    main()
