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


# vertical_hist = real_src.shape[0] - np.sum(real_src, axis=0, keepdims = True)/255
# print("verti",str(vertical_hist))
# minmax_scaler = MinMaxScaler()
# vertical_hist = vertical_hist.reshape(-1,1)
# print(vertical_hist.shape)
# minmax_scaler.fit(vertical_hist)
# minmax_scaled = minmax_scaler.fit_transform(vertical_hist)
# minmax_scaled = minmax_scaled * 10
# minmax_scaled = minmax_scaled 
# length = len(minmax_scaled)
# mean_val = np.mean(minmax_scaled)
# print(type(minmax_scaled))
# # plt.plot(minmax_scaled)
# # plt.show()

horizontal_hist = real_src.shape[1] - np.sum(real_src, axis=1,keepdims=True)/255
hist_minmax = MinMaxScaler()
horizontal_hist = horizontal_hist.reshape(-1, 1)
print("hor", horizontal_hist)
hist_minmax.fit(horizontal_hist)
hist_scaled = hist_minmax.fit_transform(horizontal_hist)
hist_scaled = hist_scaled * 10
hist_scaled = hist_scaled//2 *2
hist_mean_val = np.mean(hist_scaled)

plt.plot(hist_scaled)
plt.show()
hist_length = len(hist_scaled) 
print(hist_length)

hist_index = []
for i in range(hist_length):
    if hist_length * 0.1 < i < hist_length * 0.9:
        if hist_scaled[i] > hist_mean_val:
            hist_index.append(i)
print(hist_index)
print(hist_index[0], hist_index[-1])
# print(scaled_index)
# error = int(length*0.03)
# error_list = [0]
# for index in range(len(scaled_index)):
#     if scaled_index[index] - scaled_index[index-1] > error:
#         error_list.append(index)
# error_list.append(len(scaled_index) - 1)
# print(error_list)
# print(w)
# real_index = []

# for i in error_list:
#     real_index.append(scaled_index[i])
# # print(real_index)
# print("error = " + str(error))
# print("error_list = "+ str(error_list))
# # for i in error_list:
# #     print(i)
# # print("사용높이",h*0.1)
# real_src = cv2.cvtColor(real_src, cv2.COLOR_GRAY2BGR)
# # print(scaled_index[-1])
# # rect_img = cv2.rectangle(real_src, (int(scaled_index[0] - w*0.02), int(h*0.1)), (int(scaled_index[error_list[1]-1] + w*0.02), int(h*0.9)), (0, 0, 255), 1)
# # rect_img = cv2.rectangle(real_src, (int(scaled_index[0] ), int(h*0.1)), (int(scaled_index[error_list[1]-1] ), int(h*0.9)), (0, 0, 255), 1)
# yolo_index = []
# for i in range(len(error_list)):
#     if i == len(error_list) - 1:
#         pass
#     else :
#         # yolo_index.append(abs(int(scaled_index[error_list[i]] - w*0.02) - int(scaled_index[error_list[i+1] - 1] + w*0.02))) 
#         rect_img = cv2.rectangle(real_src, (int(scaled_index[error_list[i]] - w*0.02), int(h*0.1)), (int(scaled_index[error_list[i+1]-1] + w*0.02), int(h*0.9)), (0, 0, 255), 1)



# # yolo_cx
# img_h = abs(h*0.1 - h*0.9)
# yolo_cy = (img_h/2)/h
# yolo_h = img_h/h

# # for i 
#     # rect_img = cv2.rectangle(real_src, (int(scaled_index[i] - w*0.02), int(h*0.1)), (int(scaled_index[error_list[0]-1] + w*0.02), int(h*0.9)), (0, 0, 255), 1)


# cv2.imshow("rect", rect_img)
# cv2.waitKey(0)