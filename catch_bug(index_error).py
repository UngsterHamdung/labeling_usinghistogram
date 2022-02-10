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

class pt():
    def __init__ (self, img, pts):
        (x, y, w, h) = cv2.boundingRect(pts)
        self.pt1 = (x,y)
        self.pt2 = (x + w, y + h)
        cv2.rectangle(img, self.pt1, self.pt2, (0, 255, 0), 2)
        self.dic_x =  self.pt2[0] - self.pt1[0]
        self.dic_y =  self.pt2[1] - self.pt1[1] 

base_dir = "D:\\croped"
save_dir = "D:\\real_crop"
img_list = os.listdir(base_dir)
count = 0
img = "02우3618.jpg"
# print("현재 보고 있는 이미지 = ",str(img))

path_img = os.path.join(base_dir, img)
src_array = np.fromfile(path_img, np.uint8)
real_src = cv2.imdecode(src_array, cv2.COLOR_BGR2GRAY)
h, w = real_src.shape
cv2.imshow("asd",real_src)
cv2.waitKey(0)
mtrx = cv2.getRotationMatrix2D((w/2, h/2), 4, 1)
real_src = cv2.warpAffine(real_src, mtrx, (w,h))

# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
# dilated = cv2.dilate(real_src, kernel,iterations = 1)
contours, hierarchy = cv2.findContours(real_src, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
print("컨투어", str(contours))
temp_rect_x = []
temp_rect_y = []
temp_pt = []
for cont in contours:
    approx = cv2.approxPolyDP(cont, cv2.arcLength(cont, True) * 0.02, True)
    vtc = len(approx)
    if vtc == 4 and len(cont):
        real_src = cv2.cvtColor(real_src, cv2.COLOR_GRAY2BGR)
        rect = pt(real_src, cont)
            # print(rect.pt1, rect.pt2)
            # print(rect.pt2[0])
        
        temp_rect_x.append(rect.dic_x)
        temp_rect_y.append(rect.dic_y)
        temp_pt.append(rect.pt1)
        cor_x = min(temp_rect_x)
        cor_y = min(temp_rect_y)
        cor_xin = temp_rect_x.index(cor_x)
        realpt = temp_pt[cor_xin]
        # print(cor_x, cor_y, temp_pt, temp_rect_x )
        dst = real_src[realpt[1] : realpt[1]+temp_rect_y[cor_xin], realpt[0] : realpt[0]+cor_x].copy()
        # print(file_name)

        # cv2.imshow("crop", dst)
        # result, n = cv2.imencode(os.path.join(load_dir, file_name), dst)
        # print(result)
        # name_img = os.path.join(load_dir,file_name)
        # imwrite(name_img, dst, params=None)
        cv2.imshow("original", dst)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        cv2.waitKey(0) 
        cv2.destroyAllWindows()    
    else:
        pass
# cv2.imshow("eras", real_src)
# cv2.waitKey(0)
# src = cv2.imread(path_img,cv2.IMREAD_COLOR)
real_src = cv2.cvtColor(real_src, cv2.COLOR_BGR2GRAY)
vertical_hist = real_src.shape[0] - np.sum(real_src, axis=0, keepdims = True)/255
minmax_scaler = MinMaxScaler()
vertical_hist = vertical_hist.reshape(-1,1)
# print(vertical_hist.shape)
minmax_scaler.fit(vertical_hist)
minmax_scaled = minmax_scaler.fit_transform(vertical_hist)

minmax_scaled = minmax_scaled * 10
minmax_scaled = minmax_scaled//3 *3 
mean_val = np.mean(minmax_scaled)
length = len(minmax_scaled)

scaled_index = []

horizontal_hist = real_src.shape[1] - np.sum(real_src, axis=1,keepdims=True)/255
hist_minmax = MinMaxScaler()
horizontal_hist = horizontal_hist.reshape(-1, 1)
hist_minmax.fit(horizontal_hist)
hist_scaled = hist_minmax.fit_transform(horizontal_hist)
hist_scaled = hist_scaled * 10
hist_scaled = hist_scaled//2 *2
hist_mean_val = np.mean(hist_scaled)

# print("hist_scaled는", str(hist_scaled))
hist_length = len(hist_scaled) 
# print(hist_length)

hist_index = []

for i in range(hist_length):
    if hist_length * 0.2 < i < hist_length * 0.7:
        if hist_scaled[i] > 0:
            hist_index.append(i)
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


real_score = 0

# print("scaled_index확인", str(scaled_index))
# print("error_list", str(error_list))
# print("hist_index",str(hist_index))

if len(error_list) == 8:
    for i in range(len(error_list)):
        if i == len(error_list) - 1:
            pass
        else:
            score = abs(int(scaled_index[error_list[i]] - w*0.02) - int(scaled_index[error_list[i+1]-1] + w*0.02))
            test.append(score)
    base = np.mean(test)
    for i in test:
        # print(i)
        if base * 1.5 > i:
            real_score += 1   
    # print(test)
    # print(real_score)
    if real_score == 7:
        
        for i in range(len(error_list)):
            if i == len(error_list) - 1:
                pass
            elif len(hist_index) :
                rect_img = cv2.rectangle(real_src, (int(scaled_index[error_list[i]] - w*0.02), hist_index[0]), (int(scaled_index[error_list[i+1]-1] + w*0.02), hist_index[-1]), (0, 0, 255), 1)
        # real_path = os.path.join(save_dir, img)
        # imwrite(real_path, rect_img, params=None)
        if len(hist_index):
            count += 1
            # print(rect_img.any())
            # print("현재 이미지는", str(img))
            # print("현재 y축 높이는",str(hist_index))
            # plt.plot(hist_scaled)
            # plt.show()
            cv2.imshow("rect", rect_img)
            cv2.waitKey(0)
            print("\n")
else:
    pass
print("쓸만한 이미지는 ", str(count))