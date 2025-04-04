from mode import *
import argparse

def str2bool(v):
    return v.lower() in ('true')

parser = argparse.ArgumentParser()

#Set your dataset paths
parser.add_argument("--LR_path", type=str, default="C:\\Users\\umapa\\Downloads\\Kanishka\\Main\\SRGAN-main\\DIV2K\\DIV2K_train_LR_bicubic\\X4")
parser.add_argument("--GT_path", type=str, default="C:\\Users\\umapa\\Downloads\\Kanishka\\Main\\SRGAN-main\\DIV2K\\DIV2K_train_HR\\DIV2K_train_HR")

#Model training parameters
parser.add_argument("--res_num", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=7)
parser.add_argument("--L2_coeff", type=float, default=1.0)
parser.add_argument("--adv_coeff", type=float, default=1e-3)
parser.add_argument("--tv_loss_coeff", type=float, default=0.0)
parser.add_argument("--pre_train_epoch", type=int, default=150)
parser.add_argument("--fine_train_epoch", type=int, default=200)
parser.add_argument("--scale", type=int, default=4)
parser.add_argument("--patch_size", type=int, default=24)
parser.add_argument("--feat_layer", type=str, default='relu5_4')
parser.add_argument("--vgg_rescale_coeff", type=float, default=0.006)

#Resume training settings
parser.add_argument("--fine_tuning", type=str2bool, default=True)
parser.add_argument("--generator_path", type=str, default="C:\\Users\\umapa\\Downloads\\Kanishka\\Main\\SRGAN-main\\model\\pre_trained_model_100.pt")

#settings
parser.add_argument("--in_memory", type=str2bool, default=True)
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--verbose", type=str2bool, default=True)

args = parser.parse_args()

#Run the correct mode
if args.verbose:
    print(f"Running in {args.mode} mode...")

if args.mode == 'train':
    train(args)  
elif args.mode == 'test':
    print("Starting test...")
    test(args)  
    print("Test completed!")  
elif args.mode == 'test_only':
    test_only(args)
