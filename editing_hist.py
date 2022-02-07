from audioop import minmax
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



base_dir = "D:\\croped_img_test"
img_list = os.listdir(base_dir)
testing_img = os.path.join(base_dir,"01가3711.jpg")
print("testing_img name = ",str(testing_img))
src_array = np.fromfile(testing_img, np.uint8)
real_src = cv2.imdecode(src_array, cv2.COLOR_BGR2GRAY)
h, w = real_src.shape
# cv2.imshow("img",real_src)
# cv2.waitKey(0)


vertical_hist = real_src.shape[0] - np.sum(real_src, axis=0, keepdims = True)/255
minmax_scaler = MinMaxScaler()
vertical_hist = vertical_hist.reshape(-1,1)
print(vertical_hist.shape)
minmax_scaler.fit(vertical_hist)
minmax_scaled = minmax_scaler.fit_transform(vertical_hist)
minmax_scaled = minmax_scaled * 10
minmax_scaled = minmax_scaled // 3 * 3
length = len(minmax_scaled)
# plt.plot(minmax_scaled)
# plt.show()


# for img in img_list:
#     path_img = os.path.join(base_dir, img)
#     src_array = np.fromfile(path_img, np.uint8)
#     real_src = cv2.imdecode(src_array, cv2.COLOR_BGR2GRAY)
#     # src = cv2.imread(path_img,cv2.IMREAD_COLOR)
#     h, w = real_src.shape
#     print(h, w)
#     # cv2.imshow('asd', real_src)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     # count = 0
#     # for i in range(h):
#     #     for j in range(w):
#     #         if real_src.item(i,j) < 120:
#     #             count += 1        
#     vertical_hist = real_src.shape[0] - np.sum(real_src, axis=0, keepdims = True)/255
#     # print(type(h))
#     # hist, _ = np.histogram(vertical_hist, w)
#     # plt.plot(hist)
#     # plt.show()
#     print(len(vertical_hist[0]))


scaled_index = []


for i in range(length):
    if length * 0.1 < i <length * 0.9:
        if minmax_scaled[i] >0:
            scaled_index.append(i)

print(scaled_index)
error = int(length*0.03)
error_list = [0]
for index in range(len(scaled_index)):
    if scaled_index[index] - scaled_index[index-1] > error:
        error_list.append(index)
error_list.append(len(scaled_index) - 1)
# print(error_list)
# print(w)
# real_index = []

# for i in error_list:
#     real_index.append(scaled_index[i])
# print(real_index)
print(len(scaled_index))
print("error_list = "+ str(error_list))
print(scaled_index[56])
# for i in error_list:
#     print(i)
# print("사용높이",h*0.1)
real_src = cv2.cvtColor(real_src, cv2.COLOR_GRAY2BGR)
# print(scaled_index[-1])
# rect_img = cv2.rectangle(real_src, (int(scaled_index[0] - w*0.02), int(h*0.1)), (int(scaled_index[error_list[0]-1] + w*0.02), int(h*0.9)), (0, 0, 255), 1)
yolo_index = []
for i in range(len(error_list)):
    print(i + 1)
    if i == len(error_list) - 1:
        pass
    else :
        yolo_index.append(abs(int(scaled_index[i] - w*0.02) - int(scaled_index[error_list[i+1]] + w*0.02)))

print(yolo_index)
# yolo_cx
img_h = abs(h*0.1 - h*0.9)
yolo_cy = (img_h/2)/h
yolo_h = img_h/h

# for i 
    # rect_img = cv2.rectangle(real_src, (int(scaled_index[i] - w*0.02), int(h*0.1)), (int(scaled_index[error_list[0]-1] + w*0.02), int(h*0.9)), (0, 0, 255), 1)


# cv2.imshow("rect", rect_img)
# cv2.waitKey(0)