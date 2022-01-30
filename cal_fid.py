import os, shutil


taiwan = [
    (23, 120),
    (23, 121),
    (24, 121)
]

china = [
    (26, 101),
    (26, 100),
    (26, 99),
    (26, 98),
    (29, 101),
    (29, 100),
    (29, 99),
    (28, 101),
    (28, 100),
    (28, 99),
    (28, 98),
    (27, 101),
    (27, 100),
    (27, 99),
    (27, 98),
    (29, 98)
]


hima = [
    (29, 81),
    (29, 82),
    (29, 83),
    (28, 83),
    (28, 84),
    (28, 85),
    (27, 85),
    (27, 86),
    (28, 86),
    (27, 87)
]

peru = [
    (-7, -79),
    (-8, -79),
    (-8, -78),
    (-9, -78),
    (-10, -78),
    (-11, -77),
    (-12, -77),
    (-12, -76),
    (-13, -76),
    (-14, -76),
    (-14, -75),
    (-15, -75),
    (-15, -74),
    (-16, -73),
]

arge = [
    (-48, -74),
    (-49, -74),
    (-49, -75),
    (-50, -74),
    (-50, -75),
    (-51, -73),
    (-51, -74),
    (-51, -75),
    (-52, -74),
]

cana = [
    (58, -134),
    (57, -133),
    (56, -132),
    (56, -131),
    (55, -131),
]

cord = {"taiwan": taiwan, "china":china, "hima": hima, "peru": peru, "arge": arge, "cana": cana}

region_names = {}
regions = ["taiwan", "china", "hima", "peru", "arge", "cana"]

for item in regions:
    res = []
    for lat, lon in cord[item]:
        la = "N"
        lo = "E"
        if lat < 0:
            la = "S"
            lat *= -1
        if lon < 0:
            lo = "W"
            lon *= -1
        res.append(la+str(lat).zfill(2)+lo+str(lon).zfill(3))
    region_names[item] = res

print(region_names)

def find(name):
    name = name.split("_")
    name = name[3]
    for item in regions:
        for na in region_names[item]:
            if na == name:
                return item
    



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


for item in regions:
    clear_dir("fid/fake/"+item)
    clear_dir("fid/real/"+item)

exp_name = "test_1213"
path = "/home/jayinnn/vae-pix2pix-terrain-generator/results/{}/val_latest/images/".format(exp_name)

i = 0
for filename in sorted(os.listdir(path)):
    print(filename)
    print(find(filename))
    if filename.find("real_B") != -1:
        shutil.copyfile(path+filename, "fid/real/{}/{}.png".format(find(filename), i))
        i += 1
    elif filename.find("fake_B") != -1:
        shutil.copyfile(path+filename, "fid/fake/{}/{}.png".format(find(filename), i))


