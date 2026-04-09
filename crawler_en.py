import urllib.request
import zipfile
import os
import shutil

def download_ultimate_wikitext():
    """
    终极方案：利用 ggml (llama.cpp) 团队在 HF 上的硬核备份，通过国内镜像站极速下载
    """
    # 直接指向 ggml-org 仓库的原始 zip 文件
    url = "https://hf-mirror.com/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
    zip_path = "wikitext-2-raw.zip"
    output_file = "train_en.txt"
    
    print("🚀 绕过 AWS 拦截，正在连接 ggml-org 国内镜像节点...")
    print("⏳ 正在高速下载 WikiText-2 压缩包...")
    
    try:
        # 挂上浏览器 User-Agent，防止镜像站的安全策略拦截裸 urllib 请求
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        
        # 开启网络流与文件流，进行数据灌装
        with urllib.request.urlopen(req, timeout=20) as response, open(zip_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            
        print("✅ 下载完成！正在解压提取核心训练集...")
        
        # 从 ZIP 中精准提取目标文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            inner_file_path = "wikitext-2-raw/wiki.train.raw"
            with zip_ref.open(inner_file_path) as inner_file:
                with open(output_file, 'wb') as out_file:
                    out_file.write(inner_file.read())
                    
        # 阅后即焚，打扫战场
        os.remove(zip_path)
        
        # 统计战果
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        with open(output_file, 'r', encoding='utf-8') as f:
            char_count = len(f.read())
            
        print(f"🎉 提取成功！纯净英文语料已就绪: {output_file}")
        print(f"📊 数据规模: {file_size_mb:.2f} MB | 总字符数: {char_count:,} 个")
        print(f"🔥 AutoDL 环境就绪，去把 5090 的算力拉满吧！")
        
    except Exception as e:
        print(f"❌ 发生致命错误: {e}")

if __name__ == "__main__":
    download_ultimate_wikitext()