import cv2
import numpy as np
import math
import os

# 進行 Gamma轉換
def gamma_trans(img, gamma):  # gamma 函數處理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 顏色值為整數
    return cv2.LUT(img, gamma_table)  # 圖片顏色查表，另外可以根據光強（顏色）均勻化原則設計自適應算法。

def nothing(x):
    pass


# Source Image
read_source_image='./20220616_Data/Data_OK/0616-15-28-37-76-1.jpg'
img_source = cv2.imread(read_source_image)


# Target Image
read_target_image='./20220616_Data/Data_OK/0616-15-30-09-94-1.jpg'
img_target = cv2.imread(read_target_image)


img_gray = cv2.imread(read_source_image, 0)  # 灰階圖，用於计算gamma值


mean = np.mean(img_gray)
gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式計算gamma
image_gamma_correct = gamma_trans(img_source, gamma_val)    # gamma 變換



img_resource_resize = cv2.resize(img_source,None,fx=0.1,fy=0.1)
cv2.imshow('Source',img_resource_resize)
cv2.waitKey(0)


img_gamma_resize = cv2.resize(image_gamma_correct,None,fx=0.1,fy=0.1)
cv2.imshow('Gamma Transfer',img_gamma_resize)
key = cv2.waitKey(0)

# 按空白鍵
if key == 32:   # ASCII Code
  cv2.destroyAllWindows()
# 按's'存圖
elif key == ord('s'):
  cv2.imwrite('output.jpg',image_gamma_correct)
  cv2.destroyAllWindows()


