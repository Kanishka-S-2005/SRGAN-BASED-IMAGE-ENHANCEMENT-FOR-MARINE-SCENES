import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from losses import TVLoss, perceptual_loss
from dataset import *
from srgan_model import Generator, Discriminator
from vgg19 import vgg19
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def train(args):
    assert hasattr(args, 'scale'), "Missing argument: scale"
    assert hasattr(args, 'GT_path'), "Missing argument: GT_path"
    assert hasattr(args, 'LR_path'), "Missing argument: LR_path"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('./model', exist_ok=True)  # Ensure model directory exists

    transform = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])
    dataset = mydata(GT_path=args.GT_path, LR_path=args.LR_path, in_memory=args.in_memory, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num, scale=args.scale).to(device)

    if args.fine_tuning:
        print(f"Loading model from {args.generator_path}...")
        generator.load_state_dict(torch.load(args.generator_path, map_location=device, weights_only=True))
        print("Model loaded successfully!")  

    generator.train()
    
    l2_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr=1e-4)

    pre_epoch = 0
    while pre_epoch < args.pre_train_epoch:
        for tr_data in loader:
            gt, lr = tr_data['GT'].to(device), tr_data['LR'].to(device)
            output, _ = generator(lr)
            loss = l2_loss(gt, output)

            g_optim.zero_grad()
            loss.backward()
            g_optim.step()

        if pre_epoch % 2 == 0:
            print(f"Pre-Train Epoch {pre_epoch}: Loss = {loss.item():.6f}")

        if pre_epoch % 800 == 0:
            model_path = f'./model/pre_trained_model_{pre_epoch:03d}.pt'
            torch.save(generator.state_dict(), model_path)
            print(f"Checkpoint saved: {model_path}")

        pre_epoch += 1

def test(args):
    print("Initializing test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = mydata(GT_path=args.GT_path, LR_path=args.LR_path, in_memory=False, transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num).to(device)
    
    print(f"Loading model from {args.generator_path}...")
    generator.load_state_dict(torch.load(args.generator_path, map_location=device, weights_only=True))
    print("Model loaded successfully!")  

    generator.eval()

    os.makedirs('./result', exist_ok=True)
    psnr_list = []

    with torch.no_grad():
        for i, te_data in enumerate(loader):
            gt, lr = te_data['GT'].to(device), te_data['LR'].to(device)
            output, _ = generator(lr)

            output = output[0].cpu().numpy().clip(-1, 1)
            gt = gt[0].cpu().numpy()

            output = ((output + 1) / 2).transpose(1, 2, 0)
            gt = ((gt + 1) / 2).transpose(1, 2, 0)

            y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]

            psnr = compare_psnr(y_output, y_gt, data_range=255.0)
            psnr_list.append(psnr)

            image_path = f'./result/res_{i:04d}.png'
            Image.fromarray((output * 255).astype(np.uint8)).save(image_path)
            print(f"Saved result: {image_path} | PSNR: {psnr:.4f}")

    print(f'Average PSNR: {np.mean(psnr_list):.4f}')
    print("Test process completed.")