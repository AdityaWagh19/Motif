import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def download_file(url: str, dest_path: str):
    if not os.path.exists(dest_path):
        log.info(f"Downloading {url} to {dest_path}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                out_file.write(response.read())
            log.info(f"Successfully downloaded {dest_path}")
        except Exception as e:
            log.error(f"Failed to download {url}: {e}")
    else:
        log.info(f"File already exists: {dest_path}")

def main():
    base_dir = "benchmark_corpus"
    os.makedirs(base_dir, exist_ok=True)
    
    files_to_download = {
        "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
        "llama_paper.pdf": "https://arxiv.org/pdf/2302.13971.pdf",
        "jfk_speech.wav": "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav",
        "paddleocr_sample.jpg": "https://github.com/PaddlePaddle/PaddleOCR/raw/release/2.7/doc/imgs/11.jpg"
    }
    
    for filename, url in files_to_download.items():
        dest_path = os.path.join(base_dir, filename)
        download_file(url, dest_path)
        
if __name__ == "__main__":
    main()
