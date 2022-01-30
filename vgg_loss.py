import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage import io


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

VGG = VGGPerceptualLoss()
exp_name = "test_1213"
path = "/home/jayinnn/vae-pix2pix-terrain-generator/results/{}/val_latest/images/".format(exp_name)


path = "./"


n = len(os.listdir(path+"real"))

print(n)
# print(exp_name)

res = 0

fake = ""
real = ""

transform = transforms.Compose([transforms.ToTensor()])


for filename in tqdm(sorted(os.listdir(path+"real"))):
    real = transform(Image.open(path+"real/"+filename)).unsqueeze(0)
    fake = transform(Image.open(path+"fake/"+filename)).unsqueeze(0)
    s = VGG.forward(real, fake)
        # print(s)
    res += s

print("Average VGG Loss:", res / n)
# print("Average SSIM:", res / n)

# for filename in tqdm(sorted(os.listdir(path))):    
#     # print(filename)
#     if filename.find("real_B") != -1:
#         real = transform(Image.open(path+filename)).unsqueeze(0)
#         s = VGG.forward(real, fake)
#         res += s
#     elif filename.find("fake_B") != -1:
#         fake = transform(Image.open(path+filename)).unsqueeze(0)






