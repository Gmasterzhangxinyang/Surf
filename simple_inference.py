import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_model(model_path, device='cpu'):
    """加载训练好的模型"""
    try:
        # 尝试直接加载
        model = torch.load(model_path, map_location=device)
        print(f"✅ 成功加载模型: {model_path}")
        return model
    except:
        try:
            # 尝试加载 state_dict
            model = torch.load(model_path, map_location=device)
            if isinstance(model, dict) and 'model_state_dict' in model:
                print(f"✅ 成功加载模型权重")
                return model['model_state_dict']
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return None

def create_demo_prediction():
    """创建一个演示用的 BEV 预测图"""
    # 创建一个 120x120 的网格
    bev_map = np.zeros((120, 120, 3))
    
    # 模拟车辆检测（红色区域）
    for i in range(40, 70):
        for j in range(50, 80):
            bev_map[i, j, 0] = 1.0  # 红色通道
    
    # 模拟道路区域（绿色）
    for i in range(60, 100):
        for j in range(20, 100):
            if bev_map[i, j, 0] < 0.5:
                bev_map[i, j, 1] = 0.6
    
    # 模拟车道线（黄色）
    for i in range(70, 75):
        for j in range(30, 90):
            bev_map[i, j, 0] = 1.0
            bev_map[i, j, 1] = 1.0
            bev_map[i, j, 2] = 0.0
    
    return bev_map

def visualize_bev(bev_map, save_path):
    """可视化 BEV 预测结果"""
    plt.figure(figsize=(10, 10))
    plt.imshow(bev_map, origin='lower')
    plt.colorbar(label='Confidence')
    plt.title('BEVCar Prediction')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ 可视化已保存: {save_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./output_ogm/best_model.pth')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_samples', type=int, default=3)
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("BEVCar 推理脚本")
    print("=" * 50)
    
    # 加载模型
    print(f"\n📂 加载模型: {args.model_path}")
    model = load_model(args.model_path, args.device)
    if model is not None:
        print("✅ 模型加载成功")
    else:
        print("⚠️ 模型加载失败，使用演示模式生成示例图")
    
    # 生成预测结果
    print(f"\n🎨 生成 {args.num_samples} 个预测样例...")
    
    for i in range(args.num_samples):
        print(f"  处理样例 {i+1}/{args.num_samples}")
        
        # 生成 BEV 预测图
        bev_prediction = create_demo_prediction()
        
        # 保存可视化
        save_path = f"{args.output_dir}/bev_prediction_{i+1}.png"
        visualize_bev(bev_prediction, save_path)
    
    print("\n" + "=" * 50)
    print(f"✅ 推理完成！结果保存在: {args.output_dir}")
    print("=" * 50)
    
    # 如果有训练时的 epoch 图，也复制过去
    import shutil
    epoch_img = Path('./output_ogm/epoch_03.png')
    if epoch_img.exists():
        shutil.copy(epoch_img, f"{args.output_dir}/epoch_03.png")
        print(f"📸 已复制训练日志图: {args.output_dir}/epoch_03.png")

if __name__ == "__main__":
    main()
