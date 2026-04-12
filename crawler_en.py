import urllib.request
import zipfile
import os
import shutil

def download_ultimate_wikitext():

    url = "https://hf-mirror.com/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
    zip_path = "wikitext-2-raw.zip"
    output_file = "train_en.txt"
    

    
    try:
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
                    

        os.remove(zip_path)
        

        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        with open(output_file, 'r', encoding='utf-8') as f:
            char_count = len(f.read())
            

        print(f"就绪")
        
    except Exception as e:
        print(f"发生致命错误: {e}")

if __name__ == "__main__":
    download_ultimate_wikitext()
