import os
import cv2
import csv
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8x-pose-p6.pt')

# Open the video file
dataPath = "../data/Hit/"
imgList = os.listdir(dataPath)
imgList = sorted(imgList)

csvFile = '../The_Demo_529/test_rand.csv'
df = pd.read_csv(csvFile)
hitframe = df['HitFrame'].tolist()
landingx, landingy = df['LandingX'].tolist(), df['LandingY'].tolist()

ct = 0
for ii in imgList:
    ip = dataPath + ii
    img = cv2.imread(ip)
    res = model(img, imgsz=1280, conf=0.5, max_det=2)
    res_plotted = res[0].plot()

    # 5 keypoints for the spine, 4 keypoints for the left arm, 4 keypoints for the right arm, 2 keypoints for the left leg, and 2 keypoints for the right leg.
    keypoint_data = res[0].keypoints.cpu().detach().numpy()
    #print(keypoint_data)
    try:
        x1l, y1l, x1r, y1r = round(keypoint_data[0][-1][0]), round(keypoint_data[0][-1][1]), round(keypoint_data[0][-2][0]), round(keypoint_data[0][-2][1])
        x2l, y2l, x2r, y2r = round(keypoint_data[1][-1][0]), round(keypoint_data[1][-1][1]), round(keypoint_data[1][-2][0]), round(keypoint_data[1][-2][1])
        res_plotted = cv2.circle(res_plotted, (x1l, y1l), 5, (0,0,0), -1)
        res_plotted = cv2.circle(res_plotted, (x1r, y1r), 5, (0,0,0), -1)
        res_plotted = cv2.circle(res_plotted, (x2l, y2l), 5, (0,0,0), -1)
        res_plotted = cv2.circle(res_plotted, (x2r, y2r), 5, (0,0,0), -1)
        
        # 距離羽球最近的兩個頂點
        cx1, cy1 = (x1l + x1r) // 2, (y1l + y1r) // 2
        cx2, cy2 = (x2l + x2r) // 2, (y2l + y2r) // 2
        ballx, bally = landingx[ct], landingy[ct]
        l1 = np.sqrt((cx1-int(ballx))**2 + (cy1-int(bally))**2)
        l2 = np.sqrt((cx2-int(ballx))**2 + (cy2-int(bally))**2)

        # 找出打擊者
        if l1 < l2:

            # 找出比較進的腳
            d1 = np.sqrt((x1l-ballx)**2 + (y1l-bally)**2)
            d2 = np.sqrt((x1r-ballx)**2 + (y1r-bally)**2)
            if d1 < d2:
                df['HitterLocationX'][ct], df['HitterLocationY'][ct] = x1l, y1l
            else:
                df['HitterLocationX'][ct], df['HitterLocationY'][ct] = x1r, y1r

            # 防守者
            d1 = np.sqrt((x2l-ballx)**2 + (y2l-bally)**2)
            d2 = np.sqrt((x2r-ballx)**2 + (y2r-bally)**2)
            if d1 < d2:
                df['DefenderLocationX'][ct], df['DefenderLocationY'][ct] = x2l, y2l
            else:
                df['DefenderLocationX'][ct], df['DefenderLocationY'][ct] = x2r, y2r

        else:
            d1 = np.sqrt((x2l-ballx)**2 + (y2l-bally)**2)
            d2 = np.sqrt((x2r-ballx)**2 + (y2r-bally)**2)
            if d1 < d2:
                df['HitterLocationX'][ct], df['HitterLocationY'][ct] = x2l, y2l
            else:
                df['HitterLocationX'][ct], df['HitterLocationY'][ct] = x2r, y2r

            d1 = np.sqrt((x1l-ballx)**2 + (y1l-bally)**2)
            d2 = np.sqrt((x1r-ballx)**2 + (y1r-bally)**2)
            if d1 < d2:
                df['DefenderLocationX'][ct], df['DefenderLocationY'][ct] = x1l, y1l
            else:
                df['DefenderLocationX'][ct], df['DefenderLocationY'][ct] = x1r, y1r
    except:
        pass

    #cv2.imwrite('./predict/'+str(ct)+'.jpg', res_plotted)
    ct += 1

df.to_csv('./submit.csv', index=False)