import cv2

def imgRead(imgPath, flag=cv2.IMREAD_COLOR, resizeWidth=None, resizeHeight=None):
    img = cv2.imread(imgPath, flag)

    if resizeWidth and resizeHeight:
        img = cv2.resize(img, dsize=(resizeWidth, resizeHeight), interpolation=cv2.INTER_AREA)
        
    return img