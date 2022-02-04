from audioop import minmax
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



base_dir = "D:\\croped"
img_list = os.listdir(base_dir)
testing_img = os.path.join(base_dir,"01가3711.jpg")
src_array = np.fromfile(testing_img, np.uint8)
real_src = cv2.imdecode(src_array, cv2.COLOR_BGR2GRAY)
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
plt.plot(minmax_scaled)
plt.show()

scaled_index = []


for i in range(length):
    if length * 0.1 < i <length * 0.9:
        if minmax_scaled[i] >0:
            scaled_index.append(i)

print(scaled_index)
error = int(length*0.03)
error_list = []

for index in range(len(scaled_index)):
    if scaled_index[index] - scaled_index[index-1] > error:
        error_list.append(index)
print(error_list)
real_index = []

for i in error_list:
    real_index.append(scaled_index[i])
print(real_index)
