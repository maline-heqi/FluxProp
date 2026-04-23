import torch
from LNNModel import FluxPropLanguageModel, MarkovianGradientProbe

def main():
    # 1. 自动检测硬件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ 当前计算设备: {device}")

    # 2. 实例化探针和模型
    print("🚀 初始化余弦相似度探针与 FluxProp 模型...")
    probe = MarkovianGradientProbe()
    model = FluxPropLanguageModel(
        vocab_size=5000, 
        embed_dim=1024, 
        hidden_dim=1024, 
        rank=64, 
        num_layers=8, 
        device=device
    ).to(device)
    
    model.train() # 必须开启训练模式，否则张量不会保留梯度

    # 3. 构造一组序列测试数据 (模拟 sMNIST 的长序列特性)
    seq_len = 256
    print(f"🎲 正在生成模拟序列数据 (Batch=1, SeqLen={seq_len})...")
    input_ids = torch.randint(0, 5000, (1, seq_len)).to(device)

    # 4. 执行前向传播，把探针挂载进去
    print("🔄 执行连续时间动力学前向路由...")
    logits, _ = model(input_ids, probe=probe)

    # 5. 触发 BPTT (Backpropagation Through Time)
    print("🔙 触发 BPTT，探针正在截获深层梯度流...")
    # 用一个虚拟的 Loss 触发全局梯度回传
    loss = logits.sum() 
    loss.backward()

    # 6. 生成分析报告
    print("📊 计算完毕！正在生成理论对齐报告...")
    probe.plot_similarity_curve("cosine_similarity_sMNIST.png")

if __name__ == "__main__":
    main()