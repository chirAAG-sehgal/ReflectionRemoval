import torch
import torch.nn.functional as F
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from vq_model_og import VQ_models


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create and load model
    model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    model.to(device)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    if "ema" in checkpoint:
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("Please check model weight")
    
    model.load_state_dict(model_weight)
    del checkpoint

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize lists to store 'm' images and their reconstructions
    m_images = []
    reconstructed_m_images = []

    # Walk through the folder structure and process images one at a time
    for root, dirs, files in os.walk(args.input_dir):
        if not any(f in files for f in ['g.jpg', 'm.jpg', 'r.jpg']):
            continue  # Skip directories without necessary images

        # Locate 'm' image
        m_image_path = next((os.path.join(root, f) for f in files if 'm' in f), None)
        if m_image_path is None:
            continue

        # Process the 'm' image
        pil_image = (Image.open(m_image_path).convert("RGB")).resize((args.image_size, args.image_size))
        img = np.array(pil_image) / 255.0
        x = 2.0 * img - 1.0  # Normalize to [-1, 1]
        x = torch.tensor(x).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(device)  # Prepare for model input

        # Inference
        with torch.no_grad():
            latent, _, [_, _, indices] = model.encode(x)
            output = model.decode_code(indices, latent.shape)  # Output value is between [-1, 1]

        # Postprocess
        output = F.interpolate(output, size=[args.image_size, args.image_size], mode='bicubic').permute(0, 2, 3, 1)[0]
        sample = torch.clamp(127.5 * output + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

        # Save the reconstructed image
        out_path = os.path.join(args.output_dir, os.path.basename(m_image_path).replace('.jpg', '_reconstructed.jpg'))
        Image.fromarray(sample).save(out_path)
        print(f"Reconstructed image saved to {out_path}")

        # Append 'm' image and its reconstructed counterpart to respective lists
        m_images.append(img)
        reconstructed_m_images.append(sample)

    # Plot all original 'm' images on the top row and reconstructed 'm' images on the bottom row
    if m_images and reconstructed_m_images:
        fig, axes = plt.subplots(2, len(m_images), figsize=(15, 10))

        # Plot original 'm' images in the top row
        for i, ax in enumerate(axes[0]):
            ax.imshow(m_images[i])
            ax.axis('off')
            ax.set_title(f"Original 'm' {i+1}")

        # Plot reconstructed 'm' images in the bottom row
        for i, ax in enumerate(axes[1]):
            ax.imshow(reconstructed_m_images[i])
            ax.axis('off')
            ax.set_title(f"Reconstructed 'm' {i+1}")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/home/uas-dtu/Documents/chirag/Testset", help="Directory containing image subfolders")
    parser.add_argument("--output-dir", type=str, default="/home/uas-dtu/Documents/chirag/op", help="Directory to save output images")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-8")
    parser.add_argument("--vq-ckpt", type=str, default="checkpoints-300/0006300.pt", help="Checkpoint path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="Codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="Codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512, 1024], default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
