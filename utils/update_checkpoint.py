# -*- coding:utf-8 -*-
import argparse


parser = argparse.ArgumentParser(description="change class_num in pretrained checkpoint")

parser.add_argument("--class_num", type=int, default=80, help="num classes")
parser.add_argument("--save_dir", type=str, default="output", help="output checkpoint file to save")
parser.add_argument("--checkpoint", type=str, default=None, help="the pretrained checkpoint file path")

args = parser.parse_args()


if __name__ == "__main__":
    import os
    import torch


    if not args.checkpoint:
        usage='''python3 utils/update_checkpoint.py --class_num=80 --save_dir="output/" --checkpoint="output/yolov5s.pth"'''
        print("code error\nusage: ", usage)
        quit()

    nc = (5 + args.class_num) * 3
    checkpoint = torch.load(args.checkpoint)

    for k, v in checkpoint.items():
        if not k.startswith("detect"):
            continue
        if k.endswith("weight"):
            v = torch.nn.init.kaiming_normal_(torch.randn((nc, *v.shape[1:])), mode='fan_out', nonlinearity='relu')
        if k.endswith("bias"):
            v = torch.nn.init.zeros_(torch.randn((nc)))
        checkpoint[k] = v

    save_file = os.path.join(args.save_dir, args.checkpoint.split('/')[-1])
    torch.save(checkpoint, save_file)


