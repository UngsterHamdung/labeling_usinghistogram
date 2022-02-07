from audioop import minmax
import os
from pickle import TRUE
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


base_dir = "D:\\croped"
save_dir = "D:\\cropped_testing"
img_list = os.listdir(base_dir)
count = 0
for img in img_list:
    # print("현재 보고 있는 이미지 = ",str(img))
    path_img = os.path.join(base_dir, img)
    src_array = np.fromfile(path_img, np.uint8)
    real_src = cv2.imdecode(src_array, cv2.COLOR_BGR2GRAY)
    # src = cv2.imread(path_img,cv2.IMREAD_COLOR)
    h, w = real_src.shape
    vertical_hist = real_src.shape[0] - np.sum(real_src, axis=0, keepdims = True)/255
    minmax_scaler = MinMaxScaler()
    vertical_hist = vertical_hist.reshape(-1,1)
    # print(vertical_hist.shape)
    minmax_scaler.fit(vertical_hist)
    minmax_scaled = minmax_scaler.fit_transform(vertical_hist)

    minmax_scaled = minmax_scaled * 10
    minmax_scaled = minmax_scaled//3 *3 
    mean_val = np.mean(minmax_scaled)
    # print(type(minmax_scaled))

    length = len(minmax_scaled)
    scaled_index = []


    for i in range(length):
        if length * 0.1 < i <length * 0.9:
            if minmax_scaled[i] > mean_val:
                scaled_index.append(i)

    # print(scaled_index)
    error = int(length*0.03)
    error_list = [0]
    for index in range(len(scaled_index)):
        if scaled_index[index] - scaled_index[index-1] > error:
            error_list.append(index)
    error_list.append(len(scaled_index) - 1)
    real_src = cv2.cvtColor(real_src, cv2.COLOR_GRAY2BGR)
    test = []

    score = 0
    base = np.mean(test)
    if len(error_list) == 8:
        for i in range(len(error_list)):
            if i == len(error_list) - 1:
                pass
            else:
                score = abs(int(scaled_index[error_list[i]] - w*0.02) - int(scaled_index[error_list[i+1]-1] + w*0.02))
                test.append(score)
        print(len(test))
        for i in test:
            if base * 1.5 < i:
                score += 1   
                
        if score == 6:
            
            for i in range(len(error_list)):
                if i == len(error_list) - 1:
                    pass
                else :
                    rect_img = cv2.rectangle(real_src, (int(scaled_index[error_list[i]] - w*0.02), int(h*0.1)), (int(scaled_index[error_list[i+1]-1] + w*0.02), int(h*0.9)), (0, 0, 255), 1)
                    count += 1
                
            cv2.imshow("rect", rect_img)
            cv2.waitKey(0)
    else:
        pass
print("쓸만한 이미지는 ", str(count))