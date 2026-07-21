import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log(log_path):
    """解析 log 文件，提取每层 Fx/Fy 的 R² (all/seq-last, train/eval) 随 step 变化"""
    with open(log_path, 'r') as f:
        lines = f.readlines()

    # 数据结构：layers[层名] = {'Fx': {'steps':[], 'all_train':[], 'all_eval':[], 'seq_train':[], 'seq_eval':[]},
    #                         'Fy': {...}}
    layers = {}
    current_step = None
    in_probe = False
    current_layer = None

    # 正则表达式
    step_pattern = re.compile(r'Evaluating at step (\d+)')
    layer_pattern = re.compile(r'^([a-zA-Z0-9_]+):\s*$')  # 层名行，以冒号结尾
    # 匹配 Fx/Fy 行，例如: "  Fx                  : all train/eval = 0.0444/0.0409, sequence-last train/eval = 0.2556/0.2832"
    fx_pattern = re.compile(r'^\s+Fx\s+:\s+all train/eval = ([\d.-]+)/([\d.-]+),\s+sequence-last train/eval = ([\d.-]+)/([\d.-]+)')
    fy_pattern = re.compile(r'^\s+Fy\s+:\s+all train/eval = ([\d.-]+)/([\d.-]+),\s+sequence-last train/eval = ([\d.-]+)/([\d.-]+)')

    for line in lines:
        # 检测评估步数
        m_step = step_pattern.search(line)
        if m_step:
            current_step = int(m_step.group(1))
            in_probe = False  # 还未进入 probe 结果区域
            continue

        # 检测 "Linear Probe Results" 开始
        if 'Linear Probe Results' in line:
            in_probe = True
            continue

        if in_probe and current_step is not None:
            # 检测层名行
            m_layer = layer_pattern.match(line)
            if m_layer:
                current_layer = m_layer.group(1)
                if current_layer not in layers:
                    layers[current_layer] = {
                        'Fx': {'steps': [], 'all_train': [], 'all_eval': [], 'seq_train': [], 'seq_eval': []},
                        'Fy': {'steps': [], 'all_train': [], 'all_eval': [], 'seq_train': [], 'seq_eval': []}
                    }
                continue

            # 如果当前层存在，尝试匹配 Fx 和 Fy
            if current_layer is not None:
                m_fx = fx_pattern.match(line)
                if m_fx:
                    all_train, all_eval, seq_train, seq_eval = map(float, m_fx.groups())
                    layers[current_layer]['Fx']['steps'].append(current_step)
                    layers[current_layer]['Fx']['all_train'].append(all_train)
                    layers[current_layer]['Fx']['all_eval'].append(all_eval)
                    layers[current_layer]['Fx']['seq_train'].append(seq_train)
                    layers[current_layer]['Fx']['seq_eval'].append(seq_eval)
                    continue
                m_fy = fy_pattern.match(line)
                if m_fy:
                    all_train, all_eval, seq_train, seq_eval = map(float, m_fy.groups())
                    layers[current_layer]['Fy']['steps'].append(current_step)
                    layers[current_layer]['Fy']['all_train'].append(all_train)
                    layers[current_layer]['Fy']['all_eval'].append(all_eval)
                    layers[current_layer]['Fy']['seq_train'].append(seq_train)
                    layers[current_layer]['Fy']['seq_eval'].append(seq_eval)
                    continue

        # 遇到空行或非 probe 内容重置当前层？但为了保险，如果遇到下一层，会自动切换。

    # 清理没有数据的层
    layers = {k: v for k, v in layers.items() if v['Fx']['steps'] or v['Fy']['steps']}
    return layers

def plot_layer_r2(layers, output_png='layer_r2_curves.png'):
    """绘制每层的 Fx/Fy R² 曲线"""
    n_layers = len(layers)
    if n_layers == 0:
        print("No layer data found.")
        return

    # 计算子图布局（每行最多 4 个）
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (layer_name, data) in enumerate(layers.items()):
        ax = axes[idx]
        # 绘制 Fx
        if data['Fx']['steps']:
            steps = data['Fx']['steps']
            ax.plot(steps, data['Fx']['all_train'], 'b-', label='Fx all train')
            ax.plot(steps, data['Fx']['all_eval'], 'b--', label='Fx all eval')
            ax.plot(steps, data['Fx']['seq_train'], 'r-', label='Fx seq train')
            ax.plot(steps, data['Fx']['seq_eval'], 'r--', label='Fx seq eval')
        # 绘制 Fy
        if data['Fy']['steps']:
            steps = data['Fy']['steps']
            ax.plot(steps, data['Fy']['all_train'], 'g-', label='Fy all train')
            ax.plot(steps, data['Fy']['all_eval'], 'g--', label='Fy all eval')
            ax.plot(steps, data['Fy']['seq_train'], 'm-', label='Fy seq train')
            ax.plot(steps, data['Fy']['seq_eval'], 'm--', label='Fy seq eval')
        ax.set_title(layer_name)
        ax.set_xlabel('Step')
        ax.set_ylabel('R²')
        ax.legend(loc='best', fontsize=6)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for j in range(idx+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Plot saved to {output_png}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot_layer_r2_from_log.py <log_file_path> [output_png]")
        sys.exit(1)
    log_path = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) > 2 else 'layer_r2_curves.png'
    if not Path(log_path).exists():
        print(f"Log file {log_path} not found.")
        sys.exit(1)
    layers = parse_log(log_path)
    print(f"Found {len(layers)} layers.")
    plot_layer_r2(layers, out_png)