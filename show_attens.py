import csv
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import image_path  # dùng đúng mapping của bạn

# ============================================================================
# CONSTANTS - Dễ dàng chỉnh sửa
# ============================================================================

# Input/Output settings
PKL_PATH = "outputs/attens_flicker8k_1000_samples_384.pkl"  # Đường dẫn pickle file
DATASET = "flicker8k"  # Dataset name: "coco" hoặc "flicker8k"
VISUALIZE_INDEX = 20  # Index của record muốn visualize (0-based)

# Control flags
SHOW_STATISTICS = True  # Show thống kê về captions
EXPORT_CSV = False  # Xuất file CSV với tất cả captions
VISUALIZE_ATTENTION = True  # Tạo attention visualization

# Visualization settings
VIS_COLS = 5  # Số cột trong grid
VIS_DPI = 200  # DPI cho attention visualization
VIS_ALPHA_OVERLAY = 0.60  # Độ trong suốt của attention map
VIS_COLORMAP = "Reds"  # Colormap cho attention (Reds, Blues, Greys_r, etc.)
VIS_FONTSIZE = 12  # Font size cho word labels
VIS_FIGSIZE_PER_CELL = 3.2  # Kích thước mỗi cell trong grid

# Original image settings
ORIG_IMG_DPI = 150  # DPI cho original image
ORIG_IMG_FIGSIZE = (8, 8)  # Figure size cho original image

# Output paths
OUTPUT_DIR = Path("outputs/figure")
CSV_FILENAME = "predicted_captions.csv"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_pickle_results(pkl_path: Path):
    """
    Load pickle file - hỗ trợ cả 2 format:
    1. Code gốc: list of (img_id, (alpha, words))
    2. Code mới streaming: multiple pickle.dump() calls
    """
    results = []
    
    # Try loading as single list first (code gốc)
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, list):
                return data
    except:
        pass
    
    # Try streaming format (multiple dumps)
    try:
        with open(pkl_path, "rb") as f:
            while True:
                try:
                    results.append(pickle.load(f))
                except EOFError:
                    break
        return results
    except Exception as e:
        raise RuntimeError(f"Cannot load pickle file: {e}")


def overlay_attention_grid(img_pil: Image.Image, alpha: np.ndarray, words, out_path: Path):
    """
    Visualize attention weights for each word in caption.
    
    Args:
        img_pil: PIL image
        alpha: (T, H, W) attention weights for T words
        words: list[str] of length T
        out_path: where to save the visualization
    """
    img = img_pil.convert("RGB").resize((alpha.shape[-1], alpha.shape[-2]))
    img_np = np.asarray(img)

    T = len(words)
    rows = int(np.ceil(T / VIS_COLS))

    plt.figure(figsize=(VIS_COLS * VIS_FIGSIZE_PER_CELL, rows * VIS_FIGSIZE_PER_CELL))

    for t in range(T):
        ax = plt.subplot(rows, VIS_COLS, t + 1)
        ax.imshow(img_np)

        a = alpha[t]
        # Normalize to 0..1 for better visualization
        a = a - a.min()
        if a.max() > 0:
            a = a / a.max()

        ax.imshow(a, cmap=VIS_COLORMAP, alpha=VIS_ALPHA_OVERLAY)
        ax.set_title(words[t], fontsize=VIS_FONTSIZE, fontweight='bold')
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=VIS_DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved attention visualization to: {out_path}")


def export_captions_to_csv(results, csv_path: Path, dataset: str):
    """
    Export all predicted captions to CSV file.
    
    Args:
        results: list of (img_id, (alpha, words))
        csv_path: output CSV file path
        dataset: dataset name
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Header
        writer.writerow(['image_id', 'caption', 'num_words', 'image_path'])
        
        for rec in results:
            if not (isinstance(rec, tuple) and len(rec) == 2):
                continue
            
            img_id, payload = rec
            if not (isinstance(payload, tuple) and len(payload) == 2):
                continue
            
            alpha, words = payload
            
            # Convert words to list
            if isinstance(words, (list, tuple)):
                wlist = list(words)
            else:
                wlist = [str(words)]
            
            caption = ' '.join(wlist)
            num_words = len(wlist)
            
            try:
                img_path = image_path(int(img_id), dataset=dataset)
            except:
                img_path = "N/A"
            
            writer.writerow([img_id, caption, num_words, img_path])
    
    print(f"✓ Exported {len(results)} captions to: {csv_path}")


def print_statistics(results):
    """Print statistics about the results."""
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    
    lens = []
    word_freq = Counter()
    
    for rec in results:
        if not (isinstance(rec, tuple) and len(rec) == 2):
            continue
        
        img_id, payload = rec
        if not (isinstance(payload, tuple) and len(payload) == 2):
            continue
        
        alpha, words = payload
        
        # Convert words to list
        if isinstance(words, (list, tuple)):
            wlist = list(words)
        else:
            wlist = [str(words)]
        
        lens.append(len(wlist))
        word_freq.update(wlist)
    
    lens_np = np.array(lens, dtype=np.int32)
    print(f"📊 Total records: {len(results)}")
    print(f"📏 Caption length: min={lens_np.min()} "
          f"mean={lens_np.mean():.2f} median={np.median(lens_np):.0f} max={lens_np.max()}")
    print(f"📚 Vocabulary size: {len(word_freq)}")
    print("\n🔤 Top 30 most frequent words:")
    for w, c in word_freq.most_common(30):
        print(f"   {w:>20s} : {c:>4d}")


def visualize_record(rec, index: int, dataset: str):
    """Visualize a single record."""
    if not (isinstance(rec, tuple) and len(rec) == 2):
        print(f"❌ Invalid record format at index {index}")
        return False
    
    img_id, payload = rec
    
    if not (isinstance(payload, tuple) and len(payload) == 2):
        print(f"❌ Invalid payload format at index {index}")
        return False
    
    alpha, words = payload
    
    # Convert words to list
    if isinstance(words, (list, tuple)):
        wlist = list(words)
    else:
        wlist = [str(words)]
    
    # Convert alpha to numpy
    try:
        if hasattr(alpha, "detach"):  # torch tensor
            alpha_np = alpha.detach().cpu().numpy()
        elif hasattr(alpha, "numpy"):  # torch tensor without grad
            alpha_np = alpha.numpy()
        else:
            alpha_np = np.array(alpha)
    except Exception as e:
        print(f"❌ Cannot convert alpha to numpy: {e}")
        return False
    
    print(f"📷 Image ID: {img_id}")
    print(f"💬 Caption: \"{' '.join(wlist)}\"")
    print(f"📐 Alpha shape: {alpha_np.shape}")
    
    # Get image path
    try:
        img_path = image_path(int(img_id), dataset=dataset)
        print(f"📂 Image path: {img_path}")
    except Exception as e:
        print(f"❌ Error getting image path: {e}")
        return False
    
    if not Path(img_path).exists():
        print(f"❌ Image file not found: {img_path}")
        return False
    
    # Load image
    try:
        img = Image.open(img_path).convert("RGB")
        print(f"✓ Loaded image: {img.size}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return False
    
    # Save attention visualization
    out_png = OUTPUT_DIR / f"att_index{index}_id{img_id}.png"
    try:
        overlay_attention_grid(img, alpha_np, wlist, out_png)
    except Exception as e:
        print(f"❌ Error creating attention visualization: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save original image
    try:
        plt.figure(figsize=ORIG_IMG_FIGSIZE)
        plt.imshow(img)
        plt.title(f"Image ID: {img_id}\nCaption: {' '.join(wlist)}", fontsize=10)
        plt.axis("off")
        
        orig_png = OUTPUT_DIR / f"orig_index{index}_id{img_id}.png"
        orig_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(orig_png, dpi=ORIG_IMG_DPI, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved original image to: {orig_png}")
    except Exception as e:
        print(f"⚠️  Could not save original image: {e}")
    
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    pkl_path = Path(PKL_PATH)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

    print("=" * 60)
    print("ATTENTION VISUALIZATION TOOL")
    print("=" * 60)
    print(f"📂 Pickle file: {pkl_path}")
    print(f"📊 Dataset: {DATASET}")
    
    # Load all results
    print("\n🔄 Loading pickle results...")
    try:
        results = load_pickle_results(pkl_path)
        print(f"✓ Loaded {len(results)} records")
    except Exception as e:
        print(f"❌ Error loading pickle: {e}")
        return

    if len(results) == 0:
        print("❌ No records found in pickle file")
        return

    # Export to CSV if enabled
    if EXPORT_CSV:
        csv_path = pkl_path.parent / CSV_FILENAME
        print(f"\n📝 Exporting captions to CSV...")
        export_captions_to_csv(results, csv_path, DATASET)

    # Show statistics if enabled
    if SHOW_STATISTICS:
        print_statistics(results)

    # Visualize if enabled
    if VISUALIZE_ATTENTION:
        # Validate index
        if VISUALIZE_INDEX >= len(results):
            print(f"\n❌ Index {VISUALIZE_INDEX} out of range (max: {len(results)-1})")
            return

        # Visualize selected record
        print("\n" + "=" * 60)
        print(f"VISUALIZING RECORD #{VISUALIZE_INDEX}")
        print("=" * 60)
        
        success = visualize_record(results[VISUALIZE_INDEX], VISUALIZE_INDEX, DATASET)
        
        if success:
            print("\n" + "=" * 60)
            print("✓ Visualization complete!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ Visualization failed!")
            print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✓ Processing complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()