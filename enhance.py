import torch
import argparse
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from srgan_model import Generator
import sys
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0)

def resize_image(image, max_size=800):
    width, height = image.size
    if width > height:
        new_width, new_height = max_size, int((height / width) * max_size)
    else:
        new_width, new_height = int((width / height) * max_size), max_size
    return image.resize((new_width, new_height))

def enhance_image(model_path, image_path, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=16, scale=4).to(device)
    model_weights = torch.load(model_path, map_location=device)
    generator.load_state_dict(model_weights, strict=False)
    generator.eval()

    image = load_image(image_path).to(device)
    with torch.no_grad():
        enhanced_image, _ = generator(image)

    enhanced_image = enhanced_image.squeeze(0).cpu().numpy().clip(-1, 1)
    enhanced_image = ((enhanced_image + 1) / 2).transpose(1, 2, 0) * 255
    enhanced_image = Image.fromarray(enhanced_image.astype(np.uint8))
    resized_image = resize_image(enhanced_image)

    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.join(output_folder, f"enhanced_{os.path.basename(image_path)}")
    resized_image.save(output_filename)

    input_image_resized = resize_image(Image.open(image_path))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(input_image_resized), ax[0].set_title("Input Image"), ax[0].axis('off')
    ax[1].imshow(resized_image), ax[1].set_title("Enhanced Image"), ax[1].axis('off')
    plt.show()

    print(f"Enhanced image saved to {output_filename}")

if __name__ == "__main__":
    if 'ipykernel' in sys.argv[0]:
        sys.argv = [
            'ipykernel_launcher.py',
            '--model', r"C:\Users\umapa\Downloads\Kanishka\Main\SRGAN-main\model\pre_trained_model_100.pt",
            '--input', r"C:\Users\umapa\Downloads\Kanishka\Main\SRGAN-main\input_ei\input02.png",
            '--output', r"C:\Users\umapa\Downloads\Kanishka\Main\SRGAN-main\output_ei"
        ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    enhance_image(args.model, args.input, args.output)
