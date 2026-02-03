import json
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. 路径配置
# =========================
state_path = "New_qwen25_ceramic_lora/checkpoint-303/trainer_state.json"
save_path = "lora_eval_loss_final.png"

# =========================
# 2. 加载并解析数据
# =========================
with open(state_path, 'r', encoding='utf-8') as f:
    trainer_state = json.load(f)

eval_steps = []
eval_losses = []

for log in trainer_state['log_history']:
    if 'eval_loss' in log:
        eval_steps.append(log['step'])
        eval_losses.append(log['eval_loss'])

eval_steps = np.array(eval_steps)
eval_losses = np.array(eval_losses)

# =========================
# 3. 绘图配置（论文风格）
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

fig, ax = plt.subplots(figsize=(8, 4))

# 绘制完整验证集损失曲线
ax.plot(eval_steps, eval_losses, color='#32CD32', linewidth=1.8, alpha=0.9, label='Validation Loss')
ax.scatter(eval_steps, eval_losses, color='#FF4500', s=10, alpha=0.7)

# =========================
# 4. 设置坐标轴
# =========================
ax.set_title('LoRA Validation Loss Curve', fontsize=12, fontweight='bold')
ax.set_xlabel('Training Step', fontsize=10)
ax.set_ylabel('Loss', fontsize=10)

# X轴
ax.set_xlim(0, max(eval_steps) + 10)
ax.set_xticks(np.arange(0, max(eval_steps) + 20, 50))

# Y轴（修正：使用全部损失范围）
y_max = max(eval_losses) * 1.1
ax.set_ylim(0, y_max)
ax.set_yticks(np.linspace(0, y_max, 6))

# =========================
# 5. 图表美化
# =========================
ax.grid(True, linestyle='--', alpha=0.3, color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', frameon=False, fontsize=9)

plt.tight_layout()

# =========================
# 6. 保存图片
# =========================
plt.savefig(save_path, bbox_inches='tight', facecolor='white')
plt.close()

print(f"验证集损失图已保存至: {save_path}")