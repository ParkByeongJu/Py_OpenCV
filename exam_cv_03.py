# -*- coding: utf-8 -*-
import numpy as np
import os 
import cv2
from imgRead import imgRead
from createFolder import createFolder


# ���� �б�
img1 = imgRead("./images/iu.jpg", cv2.IMREAD_UNCHANGED, 320, 240)

# RGB ������ HSV �������� ��ȯ, ���� ���� ��ȯ
res1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

# ���� ���� ���� �� ����
res1_split = list(cv2.split(res1))
res1_split[2] = cv2.add(res1_split[2], 100)

res1_merge = cv2.merge(res1_split)
res1_merge = cv2.cvtColor(res1_merge, cv2.COLOR_HSV2BGR)

displays = [("input1", img1), ("res1", res1), ("res2", res1_split[0]), 
            ("res3", res1_split[1]), ("res4", res1_split[2]), ("res5", res1_merge)]

for(name, out) in displays:
    cv2.imshow(name, out)
    
cv2.waitKey(0)
cv2.destroyAllWindows()

save_dir='./code_res_imgs/c2_COLOR_BGR2HSV'
createFolder('./code_res_imgs/c2_COLOR_BGR2HSV')
for (name, out) in displays:
    cv2.imwrite(save_dir + "/" + name + ".jpg", out)