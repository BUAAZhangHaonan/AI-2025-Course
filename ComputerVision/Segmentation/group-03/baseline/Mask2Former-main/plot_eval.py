import json
import matplotlib.pyplot as plt
import numpy as np

def plot_training_loss(json_file_path):
    # 读取JSON文件
    with open(json_file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # 提取迭代次数和总损失

    iterations = []
    total_losses = []
    for res in data:
        if 'total_loss' in res.keys():
            iterations.append(res['iteration'])
            total_losses.append(res['total_loss'])
    
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, total_losses, 'b-', linewidth=1)
    plt.title('Training Loss Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Total Loss')
    plt.grid(True, alpha=0.3)
    
    # # 添加平滑曲线（可选）
    # window_size = 50
    # if len(total_losses) > window_size:
    #     smoothed_losses = np.convolve(total_losses, np.ones(window_size)/window_size, mode='valid')
    #     plt.plot(iterations[window_size-1:], smoothed_losses, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
    #     plt.legend()
    
    plt.tight_layout()
    # plt.show()
    plt.savefig('./loss.png')
    
    # 打印统计信息
    print(f"Total iterations: {len(iterations)}")
    print(f"Final loss: {total_losses[-1]:.4f}")
    print(f"Minimum loss: {min(total_losses):.4f} at iteration {iterations[np.argmin(total_losses)]}")
    print(f"Maximum loss: {max(total_losses):.4f} at iteration {iterations[np.argmax(total_losses)]}")



# 使用示例
if __name__ == "__main__":
    json_file_path = "./metrics.json"  # 替换为你的JSON文件路径
    plot_training_loss(json_file_path)