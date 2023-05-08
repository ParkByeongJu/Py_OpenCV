import numpy as np
import os
import cv2
from imgRead import imgRead
from createFolder import createFolder

img1 = imgRead("./images/iu.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)
img2 = imgRead("./images/iu2.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)
img3 = imgRead("./images/iu.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)
mask = np.full(shape=img3.shape, fill_value = 0, dtype = np.uint8)
h, w = img3.shape
x = (int)(w / 2) - 60; y = (int)(h / 2)- 60
cv2.rectangle(mask, (x, y), (x + 120, y + 120), (255, 255, 255), -1)

ress = []
ress.append(cv2.subtract(img1, img2))
ress.append(cv2.absdiff(img1, img2))
ress.append(cv2.bitwise_not(img3))
ress.append(cv2.bitwise_and(img3, mask)) 

displays = [("input1", img1), ("input2", img2), ("res1", ress[0]), 
            ("res2", ress[1]), ("input3", img3), ("res3", ress[2]), 
            ("res4", ress[3])]

for(name, out) in displays:
    cv2.imshow(name, out)
    
cv2.waitKey(0)
cv2.destroyAllWindows

save_dir='./code_res_imgs/c2_arithmeticLogic'
createFolder('./code_res_imgs/c2_arithmeticLogic')
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)