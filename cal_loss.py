import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from PIL import Image
from statistics import *
from torchvision import transforms
from matplotlib import pyplot as plt



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

loader = transforms.ToTensor()

unloader = transforms.ToPILImage()

def image_loader(img_path):
    img = Image.open(img_path).convert("RGB")
    img = loader(img).unsqueeze(0)
    img *= 255
    # print(img)
    # print(img.max(),  img.min())

    return img.to(device, torch.float)

def cal_loss(path1, path2):
    t1 = image_loader(path1)
    t2 = image_loader(path2)
    # print(t1.min(), t1.max())

    L1 = nn.L1Loss()
    L2 = nn.MSELoss()
    
    return (L1(t1, t2).item(), L2(t1, t2).item())


def cal_one_dir_loss(dataroot=None):
    if dataroot is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataroot", help="path to result directory.")

        args = parser.parse_args()

        dataroot = args.dataroot

    imgs = sorted(os.listdir(dataroot))
    L1 = []
    L2 = []
    cnt = 0
    for i in range(0, len(imgs), 3):
        l1, l2 = cal_loss(os.path.join(dataroot, imgs[i]), os.path.join(dataroot, imgs[i+2]))
        L1.append(l1)
        L2.append(l2)
    
    print("Average L1 Loss:", round(mean(L1), 3))
    print("Average L2 Loss:", round(mean(L2), 3))
    # print("std:", pstdev(results))

def cal_two_dir_each_size():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirA")
    parser.add_argument("--dirB")

    args = parser.parse_args()

    dirA = args.dirA
    dirB = args.dirB

    x = [i for i in range(1, 31, 2)]

    lossA = [[] for i in range(15)]
    lossB = [[] for i in range(15)]

    imgA = sorted(os.listdir(dirA))
    imgB = sorted(os.listdir(dirB))
    print(len(imgA), len(imgB))
    for i in tqdm(range(0, len(imgA), 3)):
        idx = int((int(imgA[i].split("_")[0]) - 1) / 2)
        lossA[idx].append(cal_loss(os.path.join(dirA, imgA[i]), os.path.join(dirA, imgA[i + 2])) * 100)
        lossB[idx].append(cal_loss(os.path.join(dirB, imgB[i]), os.path.join(dirB, imgB[i + 2])) * 100)

    for i in range(len(lossA)):
        lossA[i] = mean(lossA[i])
        lossB[i] = mean(lossB[i])
    
    plt.plot(x, lossA, label="modelA")
    plt.plot(x, lossB, label="modelB")

    plt.legend()
    # plt.show()
    plt.savefig("loss.png")

def cal_each_size(path):
    loss = [[] for i in range(15)]

    img = sorted(os.listdir(path))

    for i in range(0, len(img), 3):
        idx = int((int(img[i].split("_")[0]) - 1) / 2)
        loss[idx].append(cal_loss(os.path.join(path, img[i]), os.path.join(path, img[i + 2])))
    
    for i in range(len(loss)):
        loss[i] = mean(loss[i])

    return loss

def two_dir():
    # path1 = "/home/host/pytorch-CycleGAN-and-pix2pix/results/erosion/real/"
    # path2 = "/home/host/pytorch-CycleGAN-and-pix2pix/results/erosion/fake/"
    path1 = "./real/"
    path2 = "./fake/"
    L1 = []
    L2 = []
    n = len(os.listdir(path1))
    for item in os.listdir(path1):
        l1, l2 = cal_loss(path1+item, path2+item)
        L1.append(l1)
        L2.append(l2)
    
    print("Average L1 Loss:", round(mean(L1), 3))
    print("Average L2 Loss:", round(mean(L2), 3))

def cal_multi():
    dir_name = [
        "test0606_5",
        "test0606_3",
        "test0606_4",
        "test0606_2",
        "test0605",
        "test0605_2",
        "test0607",
        "test1021_hei",
        "test1021_sat",
        "china_aerial_BtoA",
        "test1021_sat_all",
        "test1021_hei_all",
        "test1022_hei",
        "test1022_sat",
        "test1022_sat_mult",
        "test1023_hei_0.5",
        "test1024_hei_0.5"
    ]

    prefix = "/home/host/pytorch-CycleGAN-and-pix2pix/results/"
    suffix = "/test_latest/images"

    for item in dir_name:
        print("{}: ".format(item))
        cal_one_dir_loss(prefix + item + suffix)


def cal_multi_plot():
    dir_name = [
        "test0606_5",
        "test0606_3",
        "test0606_4",
        "test0606_2",
        "test0605",
        "test0605_2",
        "test0607",
    ]

    prefix = "/home/host/pytorch-CycleGAN-and-pix2pix/results/"
    suffix = "test_latest/images"
    
    x = [i for i in range(1, 31, 2)]

    idx = 1
    for item in dir_name:
        print(item)
        loss = cal_each_size(os.path.join(prefix, item, suffix))
        plt.plot(x, loss, marker="o", label="model " + str(idx))
        idx += 1

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig("loss2.png")


if __name__ == "__main__":
    two_dir()
    # cal_multi()
    # cal_one_dir_loss()
