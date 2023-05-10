import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from imgRead import imgRead
from createFolder import createFolder

img1 = imgRead("./images/iu.jpg", cv2.IMREAD_GRAYSCALE, 320, 240)

res1 = cv2.equalizeHist(img1)
ch1 = [0]; ranges1 = [0,256]; histSize1 = [256]
hist1 = cv2.calcHist([img1], ch1, None, histSize1, ranges1)
hist2 = cv2.calcHist([res1], ch1, None, histSize1, ranges1)

multi_lut = np.full(shape=[256], fill_value = 0, dtype = np.uint8)
log_lut = np.full(shape=[256], fill_value = 0, dtype = np.uint8)
invol1_lut = np.full(shape=[256], fill_value = 0, dtype = np.uint8)

multi_v = 2; gamma1 = 0.4
thres1 = 5; thres2 = 100
max_v_log = 255 / np.log(1 + 255)
max_v_invol1 = 255 / np.power(255, gamma1)

for i  in range(256):
    val = i * multi_v
    if val > 255 : val = 255
    multi_lut[i] = val
    log_lut[i] = np.round(max_v_log * np.log(1 + i))
    invol1_lut[i] = np.round(max_v_invol1 * np.power(i, gamma1))
    
res2 = cv2.LUT(img1, multi_lut)
res3 = cv2.LUT(img1, log_lut)
res4 = cv2.LUT(img1, invol1_lut)
hist3 = cv2.calcHist([res2], ch1, None, histSize1, ranges1)
hist4 = cv2.calcHist([res3], ch1, None, histSize1, ranges1)
hist5 = cv2.calcHist([res4], ch1, None, histSize1, ranges1)

ress = []

ress.append(res1)
ress.append(res2)
ress.append(res3)
ress.append(res4)

displays = [("input1", img1), ("res1", ress[0]), ("res2", ress[1]),
            ("res3", ress[2]), ("res4", ress[3]), ("hist1", hist1),
            ("hist2", hist2), ("hist3",hist3), ("hist4", hist4),
            ("hist5", hist5)]

for(name, out) in displays:
    if "hist" in name:
        plt.plot(out, color='g')
        plt.xlim([0,256])
        plt.ylim([0,np.max(out)])
        plt.title(name)
        plt.show()
    else:
        cv2.imshow(name, out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

save_dir='./code_res_imgs/c2_histogramEqualization.py'
createFolder('./code_res_imgs/c2_histogramEqualization.py')
for (name, out) in displays:
    if "hist" not in name:
        cv2.imwrite(save_dir + "/" + name + ".jpg", out)