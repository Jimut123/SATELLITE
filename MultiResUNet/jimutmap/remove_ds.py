import glob
images_list = sorted(glob.glob("jimutmap_less/map/*"))
masks_list = sorted(glob.glob("jimutmap_less/roads/*"))
print(images_list[:10])
print(len(images_list))
print(len(masks_list))
print(masks_list[:10])
print(len(masks_list)- len(masks_list)*0.5)
import os

from tqdm import tqdm
"""
for img_fl in tqdm(images_list[:15488]):
    os.remove(img_fl)
    #print(img_fl)
    mask_name = "jimutmap_less/roads/"+str(img_fl.split('.')[0]).split('/')[-1]+"_road.png"
    os.remove(mask_name)
"""

