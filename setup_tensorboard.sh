#!/bin/bash

# 检查并安装 tensorboard（如果需要）
echo "正在检查 tensorboard 安装..."

# 检查是否已安装 tensorboard
python -c "import tensorboard" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ tensorboard 已安装"
else
    echo "⚠️ tensorboard 未安装，正在安装..."
    pip install tensorboard
    if [ $? -eq 0 ]; then
        echo "✅ tensorboard 安装成功"
    else
        echo "❌ tensorboard 安装失败，请手动安装: pip install tensorboard"
        exit 1
    fi
fi

echo ""
echo "📊 TensorBoard 使用说明："
echo "1. 训练时会自动在日志目录下创建 tensorboard 文件夹"
echo "2. 启动训练后，在另一个终端运行以下命令查看训练曲线："
echo "   tensorboard --logdir=logs/DHAL_SkateDog/your_experiment_name/tensorboard"
echo ""
echo "3. 然后在浏览器中打开: http://localhost:6006"
echo ""
echo "📈 记录的指标包括："
echo "   - Episode_rew/*: 每个回合的奖励"
echo "   - Loss/value_loss: 价值函数损失"
echo "   - Loss/surrogate_loss: 代理损失"
echo "   - Loss/learning_rate: 学习率"
echo "   - Perf/*: 性能指标（FPS、训练时间等）"
echo "   - Train/*: 训练指标（平均奖励、回合长度等）"
echo ""
echo "🚀 现在可以开始训练了！"