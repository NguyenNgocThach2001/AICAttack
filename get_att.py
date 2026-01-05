import json
import pickle
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from scipy.misc import imread, imresize

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import skimage.transform

from utils import *  # expects image_path(id, dataset=...)

# download the show and tell repo and put it in models/SAT
sys.path.insert(1, "models/SAT")
from models import *

# This code snippet requires Python 3.6 and PyTorch 1.6 with torchvision 0.7.0

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
warnings.filterwarnings("ignore")

CHECKPOINT_DIR = Path("checkpoint")
CHECKPOINT_FILE = CHECKPOINT_DIR / "BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
WORDMAP_FILE = CHECKPOINT_DIR / "WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figure"

DEFAULT_BEAM_SIZE = 5
DEFAULT_MAX_CAPTION_LENGTH = 50
DEFAULT_VIS_GRID_SIZE = 14
DEFAULT_VIS_UPSCALE = 24
DEFAULT_VIS_SMOOTH_SIGMA = 8
DEFAULT_VISUALIZE_IMAGE_SIZE = 255
DEFAULT_ALPHA_SIZE = 384
DEFAULT_SAMPLE_COUNT = 1000
DEFAULT_SAMPLE_RANGE = 8000


def caption_image_beam_search(encoder, decoder, word_map, image_path=None, beam_size=3):
    """
    Reads an image and captions it with beam search (ORIGINAL VERSION).
    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """
    k = beam_size
    vocab_size = len(word_map)

    # Read image and process (ORIGINAL METHOD)
    if image_path:
        img = imread(image_path)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = imresize(img, (255, 255))
        img = img.transpose(2, 0, 1)
        img = img / 255.
        img = torch.FloatTensor(img).to(device)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([normalize])
        image = transform(img)  # (3, image_size, image_size)
    else:
        image = image

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visualize_att(image_path_str, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.
    """
    image = Image.open(image_path_str).convert("RGB")
    image = image.resize(
        [DEFAULT_VIS_GRID_SIZE * DEFAULT_VIS_UPSCALE, DEFAULT_VIS_GRID_SIZE * DEFAULT_VIS_UPSCALE],
        Image.LANCZOS,
    )

    words = [rev_word_map[ind] for ind in seq]

    plt.figure(figsize=(15, 15))
    for t in range(len(words)):
        if t > DEFAULT_MAX_CAPTION_LENGTH:
            break

        plt.subplot(int(np.ceil(len(words) / 5.0)), 5, t + 1)
        plt.text(0, 1, f"{words[t]}", color="black", backgroundcolor="white", fontsize=12)
        plt.imshow(image)

        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(
                current_alpha.numpy(),
                upscale=DEFAULT_VIS_UPSCALE,
                sigma=DEFAULT_VIS_SMOOTH_SIGMA,
            )
        else:
            alpha = skimage.transform.resize(
                current_alpha.numpy(),
                [DEFAULT_VIS_GRID_SIZE * DEFAULT_VIS_UPSCALE, DEFAULT_VIS_GRID_SIZE * DEFAULT_VIS_UPSCALE],
            )

        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)

        plt.set_cmap(cm.Greys_r)
        plt.axis("off")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "attention_1.png")
    plt.close()


@torch.no_grad()
def get_alphas_with_loaded(
    img_id,
    encoder,
    decoder,
    word_map,
    rev_word_map,
    dataset="coco",
):
    """
    Use already-loaded encoder/decoder/wordmap.
    Return:
      alpha_resized: (T, DEFAULT_ALPHA_SIZE, DEFAULT_ALPHA_SIZE) tensor on CPU
      words: list[str]
    """
    img_path = image_path(img_id, dataset=dataset)

    # Fail fast if missing
    if not Path(img_path).exists():
        raise FileNotFoundError(img_path)

    seq, alphas = caption_image_beam_search(
        encoder,
        decoder,
        word_map,
        image_path=img_path,
        beam_size=DEFAULT_BEAM_SIZE,
    )

    # alphas: list -> tensor
    alphas = torch.FloatTensor(alphas)

    # Remove <start> and <end> to align with words
    alphas = alphas[1:-1]

    # Resize attention maps to DEFAULT_ALPHA_SIZE
    alpha_resized = F.interpolate(
        alphas.unsqueeze(0),
        size=(DEFAULT_ALPHA_SIZE, DEFAULT_ALPHA_SIZE),
        mode="bilinear",
    ).squeeze(0)

    words = [rev_word_map[ind] for ind in seq][1:-1]
    return alpha_resized.cpu(), words


if __name__ == "__main__":
    print("=" * 60)
    print("Image Captioning with Attention - Starting...")
    print("=" * 60)

    dataset_name = "flicker8k"
    processed_count = 0

    try:
        print(f"\n🖥️  device: {device}")

        # Load models ONCE
        print("\n📦 Loading models (once)...")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=str(device))

        decoder = checkpoint["decoder"].to(device).eval()
        encoder = checkpoint["encoder"].to(device).eval()
        print("✓ Models loaded successfully")

        # Load word map ONCE
        print("\n📖 Loading word map...")
        with open(WORDMAP_FILE, "r") as j:
            word_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}
        print("✓ Word map loaded successfully")

        # Sampling
        num_samples = DEFAULT_SAMPLE_COUNT
        random.seed(42)
        ids = random.sample(range(DEFAULT_SAMPLE_RANGE), num_samples)

        # Prepare output file
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_filename = OUTPUT_DIR / f"attens_{dataset_name}_{num_samples}_samples_{DEFAULT_ALPHA_SIZE}.pkl"
        
        print(f"\n🎯 Processing {num_samples} images from {dataset_name} dataset")
        print(f"📝 Writing results to: {output_filename}")
        print(f"Selected IDs: {ids[:10]}..." if len(ids) > 10 else f"Selected IDs: {ids}")
        print("-" * 60)

        # Collect all results first
        attens = []
        
        for idx, img_id in enumerate(ids):
            try:
                print(f"\n[{idx+1}/{num_samples}] Processing image ID: {img_id}...", end=" ", flush=True)
                alpha_resized, words = get_alphas_with_loaded(
                    img_id,
                    encoder,
                    decoder,
                    word_map,
                    rev_word_map,
                    dataset=dataset_name,
                )
                
                # Collect result
                attens.append((img_id, (alpha_resized, words)))
                processed_count += 1

                caption = " ".join(words)
                print("✓")
                print(f'  Caption: "{caption}"')
                
                # Clean GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except FileNotFoundError as e:
                print("✗ Image not found")
                print(f"  Error: {e}")
                continue
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user!")
                raise
            except Exception as e:
                print("✗ Failed")
                print(f"  Error: {e}")
                continue

        # Write all results at once
        if processed_count > 0:
            print(f"\n💾 Writing {processed_count} results to file...", end=" ", flush=True)
            with open(output_filename, "wb") as file:
                pickle.dump(attens, file)
            print("✓")

        # Final summary
        print("\n" + "=" * 60)
        if processed_count > 0:
            print(f"✓ Successfully saved {processed_count} results to:")
            print(f"  {output_filename}")
            print("\n📊 Summary:")
            print(f"  - Total processed: {processed_count}/{num_samples}")
            print(f"  - Success rate: {processed_count/num_samples*100:.1f}%")
        else:
            print("⚠️  No results to save - all images failed!")

        print("=" * 60)
        print("✓ Processing complete!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
        if processed_count > 0:
            # Save partial results
            print(f"\n💾 Saving {processed_count} partial results...", end=" ", flush=True)
            with open(output_filename, "wb") as file:
                pickle.dump(attens, file)
            print("✓")
            print(f"Saved to: {output_filename}")
        sys.exit(0)

    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)