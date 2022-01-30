import os
import numpy as np
from tqdm import tqdm

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error
from skimage import io


exp_name = "1213_hei_all"
# phase_name = "val_latest"
# path = "/home/host/pytorch-CycleGAN-and-pix2p ix/results/{}/{}/images/".format(exp_name, phase_name)

path = "./"


n = len(os.listdir(path+"real"))
print(n)
# print(exp_name)

res_ssim = 0
res_psnr = 0

fake = ""
real = ""

for filename in tqdm(sorted(os.listdir(path+"real"))):
    real = io.imread(path+"real/"+filename)
    fake = io.imread(path+"fake/"+filename)
    s = ssim(real, fake, multichannel=True, data_range=255)
    p = psnr(real, fake, data_range=255)
        # print(s)
    res_ssim += s
    res_psnr += p

print("Average SSIM:", res_ssim / n)
print(f"Average PSNR: {res_psnr/n}")

# for filename in tqdm(sorted(os.listdir(path))):
#     if filename.find("real_B") != -1:
#         real = io.imread(path+filename)
#         # print(real)
#         # print(real[0][0])
#         # print("---")
#         # print(fake[0][0])
#         # print("")
#         s = ssim(real, fake, multichannel=True, data_range=255)
#         # print(s)
#         res += s
#     elif filename.find("fake_B") != -1:
#         fake = io.imread(path+filename)

# print("Average SSIM:", res / 400)
