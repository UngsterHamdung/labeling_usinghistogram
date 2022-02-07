from audioop import minmax
import os
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
    # print("before_len_minmax_scaled = "+str(len(minmax_scaled)))
    
    # print("edited_range = ",str(int(w*0.03)))
    # edited = minmax_scaled.copy()
    # for i in range(int(w*0.04)):
    #     front_edited = np.delete(edited, i, 0)
    #     edited = front_edited
    #     print(i)
    # print("before",str(len(edited))) 
    # temp_w = len(edited)
    # for i in range(temp_w - 1, temp_w - int(w*0.04) - 1, -1):
    #     edited = np.delete(edited, i, 0)
    # print("back",str(len(edited)))
    minmax_scaled = minmax_scaled * 10
    minmax_scaled = minmax_scaled//3 *3 
    mean_val = np.mean(minmax_scaled)
    # print(type(minmax_scaled))

    length = len(minmax_scaled)
    # print("len_of_minmax_scaled = "+str(length))
    # plt.plot(minmax_scaled)
    # plt.show()



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

    # print(len(scaled_index))
    # print("error_list = "+ str(error_list))
    # x1 = scaled_index[error_list[i]] - w*0.02
    # x2 = scaled_index[error_list[i+1]-1] + w*0.02
    if len(error_list) == 8:
        # print("all_list = ", str(scaled_index))
        # print("error_list = ", str(error_list))
        # print("len_of_error = ", str(len(error_list)))
        test = []
        for i in range(len(error_list)):
            if i == len(error_list) - 1:
                pass
            else:
                score = abs(int(scaled_index[error_list[i]] - w*0.02) - int(scaled_index[error_list[i+1]-1] + w*0.02))
                test.append(score)
        print(test)
        base = np.mean(test)
        # print(base = np.mean(test))
        for i in test:
            if base * 1.5 < i:
               pass
            else:
                
                for i in range(len(error_list)):
                    if i == len(error_list) - 1:
                        pass
                    else :
                        # yolo_index.append(abs(int(scaled_index[error_list[i]] - w*0.02) - int(scaled_index[error_list[i+1] - 1] + w*0.02))) 
                        rect_img = cv2.rectangle(real_src, (int(scaled_index[error_list[i]] - w*0.02), int(h*0.1)), (int(scaled_index[error_list[i+1]-1] + w*0.02), int(h*0.9)), (0, 0, 255), 1)
                        count += 1
                        

        # rect_img = cv2.rectangle(real_src, (int(scaled_index[0] - w*0.02), int(h*0.1)), (int(scaled_index[error_list[1]-1] + w*0.02), int(h*0.9)), (0, 0, 255), 1)
        # rect_img = cv2.rectangle(real_src, (int(scaled_index[0] ), int(h*0.1)), (int(scaled_index[error_list[1]-1] ), int(h*0.9)), (0, 0, 255), 1)
        # for i 
            # rect_img = cv2.rectangle(real_src, (int(scaled_index[i] - w*0.02), int(h*0.1)), (int(scaled_index[error_list[0]-1] + w*0.02), int(h*0.9)), (0, 0, 255), 1)
        # name_img = os.path.join(save_dir, img)
        # imwrite(name_img, rect_img, params=None)
        cv2.imshow("rect", rect_img)
        cv2.waitKey(0)
    else:
        pass
print("쓸만한 이미지는 ", str(count))