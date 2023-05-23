import numpy as np
import os
import cv2
import json
from h2pose.Hfinder import Hfinder

if __name__ == '__main__':
    cap = cv2.VideoCapture('../output/1620780853.7002041predict.mp4')
    ret, img = cap.read()
    if ret:
        court2D = []
        #court3D = [[-3.05, 6.7], [3.05, 6.7], [3.05, -6.7], [-3.05, -6.7]]
        court3D = [[-3.05, 1.98], [3.05, 1.98], [3.05, -1.98], [-3.05, -1.98]]
        hf = Hfinder(img, court2D=court2D, court3D=court3D, pad=[350,350,350,350])
        Hmtx = hf.getH()
        print(Hmtx)
        with open('Hmtx_o.json', 'w') as f:
            json.dump(Hmtx.tolist(), f)
    cap.release()
