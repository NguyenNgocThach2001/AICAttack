import kagglehub
import shutil
from pathlib import Path

# Download về thư mục hiện tại
print("Downloading Flickr8k dataset...")
kaggle_path = kagglehub.dataset_download("adityajn105/flickr8k")
print(f"Downloaded to: {kaggle_path}")

# Copy về thư mục hiện tại
current_dir = Path.cwd()
target_dir = current_dir / "flickr8k"

if target_dir.exists():
    shutil.rmtree(target_dir)

shutil.copytree(kaggle_path, target_dir)
print(f"Copied to: {target_dir}")