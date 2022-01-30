import os, shutil


def clear_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


clear_dir("real")
clear_dir("fake")

exp_name = "test_0126"
path = "/home/jayinnn/vae-pix2pix-terrain-generator/results/{}/val_latest/images/".format(exp_name)

i = 0
print(len(os.listdir(path)) // 5)
print(exp_name)
for filename in sorted(os.listdir(path)):
    if filename.find("real_C") != -1:
        shutil.copyfile(path+filename, "./real/{}.png".format(i))
        i += 1
    elif filename.find("fake_C") != -1:
        shutil.copyfile(path+filename, "./fake/{}.png".format(i))
# path = "/home/host/scifair/torch_erosion/results/"
# i = 0
# for filename in sorted(os.listdir(path)):
#     shutil.copyfile(path+filename, "./fake/{}.png".format(i))
#     i += 1

# i = 0

# exp_name = "test_1213"
# phase_name = "val_latest"
# path = "/home/jayinnn/vae-pix2pix-terrain-generator/results/{}/{}/images/".format(exp_name, phase_name)

# for filename in sorted(os.listdir(path)):
#     if filename.find("real_C") != -1:
#         shutil.copyfile(path+filename, "./real/{}.png".format(i))
#         i += 1
