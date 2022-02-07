from audioop import minmax
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



base_dir = "D:\\croped_img_test"
img_list = os.listdir(base_dir)

for img in img_list:
    print(type(img))
    path_img = os.path.join(base_dir, img)
    src_array = np.fromfile(path_img, np.uint8)
    real_src = cv2.imdecode(src_array, cv2.COLOR_BGR2GRAY)
    # src = cv2.imread(path_img,cv2.IMREAD_COLOR)
    h, w = real_src.shape
    vertical_hist = real_src.shape[0] - np.sum(real_src, axis=0, keepdims = True)/255
    minmax_scaler = MinMaxScaler()
    vertical_hist = vertical_hist.reshape(-1,1)
    print(vertical_hist.shape)
    minmax_scaler.fit(vertical_hist)
    minmax_scaled = minmax_scaler.fit_transform(vertical_hist)
    minmax_scaled = minmax_scaled * 10
    minmax_scaled = minmax_scaled // 3 * 3
    length = len(minmax_scaled)


    scaled_index = []


    for i in range(length):
        if length * 0.1 < i <length * 0.9:
            if minmax_scaled[i] >0:
                scaled_index.append(i)

    error = int(length*0.03)
    error_list = [0]
    for index in range(len(scaled_index)):
        if scaled_index[index] - scaled_index[index-1] > error:
            error_list.append(index)
    error_list.append(len(scaled_index) - 1)

    print(len(scaled_index))
    print("error_list = "+ str(error_list))
    print(scaled_index[56])
    real_src = cv2.cvtColor(real_src, cv2.COLOR_GRAY2BGR)
    yolo_index = []
    for i in range(len(error_list)):
        if i == len(error_list) - 1:
            pass
        else :
            print(scaled_index[error_list[i]], scaled_index[error_list[i+1] - 1])
            yolo_index.append(abs(int(scaled_index[error_list[i]] - w*0.02) - int(scaled_index[error_list[i+1] - 1] + w*0.02))) 
    
    print(yolo_index)
    # yolo_cx
    img_h = abs(h*0.1 - h*0.9)
    yolo_cy = (img_h/2)/h
    yolo_h = img_h/h