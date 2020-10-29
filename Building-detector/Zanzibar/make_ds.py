import glob 
total_img_train = glob.glob('znz-segment-z19/znz-train-z19-all-buffered/images-512/*')
print(len(total_img_train))
total_img_train.sort()
print(total_img_train[:10])


train_len = int(len(total_img_train)*0.9)
test_len = len(total_img_train) - train_len

from tqdm import tqdm
import shutil
count =  1
print("Train split : ")
for item in tqdm(total_img_train[:train_len]):
  name = str(item.split('.')[0]).split('/')[-1][:-4]
  img_name = "znz-segment-z19/znz-train-z19-all-buffered/images-512/" +name + "_img.jpg"
  mask_name = "znz-segment-z19/znz-train-z19-all-buffered/masks-512/"+name+"_mask_buffered.png"
  if count<=train_len:
    shutil.move(img_name,'original_img_train/')
    shutil.move(mask_name,'ground_truth_train/')
  count += 1
count =  1
print("Test split : ")
for item in tqdm(total_img_train[train_len:train_len+test_len]):
  name = str(item.split('.')[0]).split('/')[-1][:-4]
  img_name = "znz-segment-z19/znz-train-z19-all-buffered/images-512/" +name + "_img.jpg"
  mask_name = "znz-segment-z19/znz-train-z19-all-buffered/masks-512/"+name+"_mask_buffered.png"
  if count<=test_len:
    shutil.move(img_name,'original_img_test/')
    shutil.move(mask_name,'ground_truth_test/')
  count += 1

