import os
import sys
import json
import subprocess
import zipfile
import shutil
from pathlib import Path

def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        download_url_file = os.path.join(subdir_path, "download_url.json")
        if not os.path.exists(download_url_file):
            continue
        
        with open(download_url_file, 'r') as f:
            url = json.load(f).get('url', '')
        
        if 'kaggle.com/datasets/' in url:
            dataset_id = url.split('kaggle.com/datasets/')[-1].strip('/')
            result = subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_id, '-p', subdir_path], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                for zip_file in Path(subdir_path).glob('*.zip'):
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(subdir_path)
                    os.remove(zip_file)
                
                for root, dirs, files in os.walk(subdir_path):
                    for file in files:
                        if file.endswith(('.csv', '.xlsx')):
                            src = os.path.join(root, file)
                            dst = os.path.join(subdir_path, file)
                            if src != dst:
                                shutil.move(src, dst)
            else:
                print(f"Failed to download {dataset_id}: {result.stderr}")

if __name__ == "__main__":
    main()

