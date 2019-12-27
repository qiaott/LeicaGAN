from torchvision import transforms
import os
import cv2 as cv
from PIL import Image
from scipy import ndimage
import numpy as np
from scipy.stats import entropy
from PIL import Image

# data = np.random.randint(0, 255, size=300)
# # print(data)
# img = data.reshape(10,10,3)
# # print(img.shape)
# img_tensor = transforms.ToTensor()(img)
# print(img_tensor)
# img_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_tensor)
# print('-')
path = '/home/qtt/AttnGAN_Code/data/coco/images'
img_names = os.listdir(path)
for i in range(1):
    img_name = img_names[i]
    img_path = os.path.join(path, img_name)
    img1 = cv.imread(img_path)
    img2 = ndimage.imread(img_path)
    img3 = np.asarray(Image.open(img_path).convert('RGB'))
    img4 =
    print(img1[0])
    print('---')
    print(img2[0])
    print('---')
    print(img3[0])
    print('---')




