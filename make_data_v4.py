import cv2
import os
import numpy as np
import random
import glob
random.seed(123)
def generate_train_dataset( save_path, slide=256, crop_size=512):
  os.makedirs(save_path, exist_ok=True)

  # Noise list
  list_train_file = open('../list_train_file_v10.txt','r').readlines()
  # Read image's pair 
  list_images = glob.glob('../Original_data/Double_2021/*.png')
  for line in list_images:
    #print(line)
    if not os.path.basename(line)+'\n' in list_train_file:
      print(os.path.basename(line))
      continue
    double_image = cv2.imread(line,0)
    h, w = double_image.shape
    raw_image = double_image[:int(h/2),:]
    gt_image = double_image[int(h/2):,:]

    h_raw, w_raw = raw_image.shape[:2]
    h_gt, w_gt = gt_image.shape[:2]
    if h_raw != h_gt or w_raw != w_gt:
      continue


    w_new = int(np.ceil(w_raw/(512))*(512))
    pad_raw = np.zeros((h_raw, w_new), dtype=np.uint8)
    pad_clean = np.zeros((h_raw, w_new), dtype=np.uint8)
    pad_raw[:, :w_raw] = raw_image
    pad_clean[:, :w_raw] = gt_image
    count = 0
    w_step = int(h_raw / 256 * 512) 
    for j in range(0,w_new-w_step,256):

      crop_raw_image = pad_raw[:,j:j+w_step]
      crop_gt_image = pad_clean[:,j:j+w_step]
      crop_raw_image = cv2.resize(crop_raw_image, (512, 256)) 
      crop_gt_image = cv2.resize(crop_gt_image, (512, 256)) 

      output_image = np.concatenate((crop_raw_image, crop_gt_image),axis=0)
      if output_image.shape[0] != 512 or  output_image.shape[1] != 512:
        print(output_image.shape)
        assert(1==2)
      cv2.imwrite(save_path + '/' + os.path.basename(line)[:-4] + '_' + str('%04d'%count)+ '.jpg', output_image)
      count = count +1
  print('_' * 50)
  print('All files have been cliped successfully!')



if __name__ =="__main__":
    save_path = '../pix2pix_data_v10full_256_512_resize'

    generate_train_dataset(save_path)