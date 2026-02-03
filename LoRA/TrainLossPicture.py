import json
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. 路径配置
# =========================
state_path = "New_qwen25_ceramic_lora/checkpoint-303/trainer_state.json"
save_path = "lora_training_loss_final.png"

# =========================
# 2. 加载并解析数据（修复KeyError核心）
# =========================
with open(state_path, 'r', encoding='utf-8') as f:
    trainer_state = json.load(f)

all_steps = []
all_losses = []

# 关键修复：先判断log里是否有'loss'字段，再添加
for log in trainer_state['log_history']:
    if 'loss' in log and 'step' in log:  # 同时校验step和loss，避免异常
        all_steps.append(log['step'])
        all_losses.append(log['loss'])

# 健壮性处理：如果没有获取到任何loss数据，给出提示并退出
if not all_losses:
    print("错误：未在日志中找到任何训练loss数据！")
    exit(1)

all_steps = np.array(all_steps)
all_losses = np.array(all_losses)

# 分离前50步和50步后的数据
start_step = 50
mask = all_steps >= start_step
plot_steps = all_steps[mask]
plot_losses = all_losses[mask]

# 健壮性处理：如果50步后无数据，使用全部数据
if len(plot_losses) == 0:
    plot_losses = all_losses
    print("提示：50步后无数据，将使用全部数据绘制Y轴范围")

# =========================
# 3. 绘图配置（论文风格）
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

fig, ax = plt.subplots(figsize=(8, 4))

# 绘制完整曲线
ax.plot(all_steps, all_losses, color='#4169E1', linewidth=1.8, alpha=0.9, label='Training Loss')
ax.scatter(all_steps, all_losses, color='#FF6347', s=10, alpha=0.7)

# =========================
# 4. 核心：Y轴只适配50步后的数据，让后期波动清晰
# =========================
ax.set_title('LoRA Training Loss Curve', fontsize=12, fontweight='bold')
ax.set_xlabel('Training Step', fontsize=10)
ax.set_ylabel('Loss', fontsize=10)

# X轴：从0到最大step
ax.set_xlim(0, max(all_steps) + 10)
ax.set_xticks(np.arange(0, max(all_steps) + 20, 50))

# Y轴：只看50步之后的范围，让波动放大
y_max = max(plot_losses) * 1.1  # 50步后的最大损失
ax.set_ylim(0, y_max)
ax.set_yticks(np.linspace(0, y_max, 6))  # 生成6个刻度，更均匀

# =========================
# 5. 图表美化
# =========================
ax.grid(True, linestyle='--', alpha=0.3, color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', frameon=False, fontsize=9)

# 紧凑布局
plt.tight_layout()

# =========================
# 6. 保存图片
# =========================
plt.savefig(save_path, bbox_inches='tight', facecolor='white')
plt.close()

print(f"优化后的损失图已保存至: {save_path}")