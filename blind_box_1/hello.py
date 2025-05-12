import swanlab
import random

# 创建一个SwanLab项目
swanlab.init(
    project="hello_project",  # 设置项目名
    experiment_name="hello",  # 设置实验名，用于在项目中快速查找本次实验
    description="这是我第一次使用swanlab",  # 对于本实验的说明
    
    # 设置超参数
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 100
    },

    # logdir="./SwanLab_logs",
    # mode="local",
)

# 模拟一次训练
offset = random.random() / 5
for epoch in range(2, swanlab.config['epochs']):
  acc = 1 - 2 ** -epoch - random.random() / epoch - offset
  loss = 2 ** -epoch + random.random() / epoch + offset

  # 记录训练指标
  swanlab.log({"acc": acc, "loss": loss})

# [可选] 完成训练，这在notebook环境中是必要的
swanlab.finish()