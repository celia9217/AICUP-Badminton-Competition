import cv2
import numpy as np
import json
from math import cos, sin, pi
np.set_printoptions(suppress=True)

class Hfinder(object):
    """docstring for Hfinder"""
    def __init__(self, img=None, court2D=[], court3D=[[-3.05, 6.7], [3.05, 6.7], [3.05, -6.7], [-3.05, -6.7]], pad=[0,0,0,0], downScale=False):
        super(Hfinder, self).__init__()
        self.pad = pad
        self.img = cv2.copyMakeBorder(img, pad[0],pad[1],pad[2],pad[3], cv2.BORDER_CONSTANT, value=[0,0,0]) # padding
        self.downScale = downScale
        if self.downScale:
            self.img = cv2.resize(self.img, (self.img.shape[1]//2, self.img.shape[0]//2))
        self.court2D = court2D
        print(type(self.court2D))
        self.court3D = court3D
        self.H = np.zeros((3,3)) # mapping 2D pixel to wcs 3D plane
        self.calculateH(self.img)

    def getH(self):
        return self.H        

    def mouseEvent(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            if len(self.court2D) < 4:
                self.court2D.append([x, y])
            else:
                idx = np.linalg.norm(np.array(self.court2D) - np.array([[x, y]]), axis=1).argmin()
                self.court2D[idx] = [x, y]

    def calculateH(self, img):
        if len(self.court2D) == 0:
            cv2.namedWindow("Please pick 4 point of court")
            cv2.setMouseCallback("Please pick 4 point of court", self.mouseEvent)
            while True:
                show = np.copy(img)
                for c in self.court2D:
                    cv2.circle(show, (c[0], c[1]), 3, (38, 28, 235), -1)
                if len(self.court2D) > 1:
                    cv2.drawContours(show, [np.array(self.court2D)], 0, (38, 28, 235), 1)
                cv2.imshow("Please pick 4 point of court", show)
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    break

            if self.downScale:
                self.court2D = np.array(self.court2D)*2 - np.array([[self.pad[2],self.pad[0]]]) # unpadding
            else:
                self.court2D = np.array(self.court2D) - np.array([[self.pad[2],self.pad[0]]]) # unpadding
        else:
            self.court2D = np.array(self.court2D)
        self.court3D = np.array(self.court3D)
        self.H, status = cv2.findHomography(self.court2D, self.court3D)


if __name__ == '__main__':
    img = cv2.imread('track_court.png')
    hf = Hfinder(img, pad=[0,0,0,0])
    H = hf.getH()
    with open('Hmtx.json', 'w') as f:
    	json.dump(H.tolist(), f)
    # H=H*np.array([[-1,-1,-1],[-1,-1,-1],[-1,-1,1]])

    tt = (H@np.array([hf.court2D[0,0], hf.court2D[0,1], 1]).reshape(3,1)).reshape(3)
    tt = tt/tt[2]
    print('Check Homo:', tt) # should output (-3.05, 6.7, 1)

    K = np.array([[989.09, 0, 1280//2],
                  [0, 989.09,  720//2],
                  [0,      0,       1]])

    K_inv = np.linalg.inv(K)
    H_inv = np.linalg.inv(H) # H_inv: wcs -> ccs
    multiple = K_inv@H_inv[:,0]
    lamda1 = np.linalg.norm(K_inv@H_inv[:,0], ord=None, axis=None, keepdims=False)
    lamda2 = np.linalg.norm(K_inv@H_inv[:,1], ord=None, axis=None, keepdims=False)
    lamda3 = (lamda1+lamda2)/2

    R = np.zeros((3,3))
    t = np.zeros(3)
    P = np.zeros((3,4))
    
    R[:,0] = (K_inv@H_inv[:,0])/lamda1
    R[:,1] = (K_inv@H_inv[:,1])/lamda2
    R[:,2] = np.cross(R[:,0], R[:,1])
    t = np.array((K_inv@H_inv[:,2])/lamda3)

    P[:,:3] = R
    P[:,3] = t

    tt = K @ P @ np.array([-3.05, 6.7, 0, 1]).reshape(4,1)
    tt = tt/tt[2]
    print('Check Proj:', tt.reshape(3)) # should output (hf.court2D[0,0], hf.court2D[0,1], 1)


    Ori_ccs = (P @ [[0],[0],[0],[1]])
    cir_pose_i_ccs = (P @ [[1],[0],[0],[1]]) - Ori_ccs
    cir_pose_j_ccs = (P @ [[0],[1],[0],[1]]) - Ori_ccs
    cir_pose_k_ccs = (P @ [[0],[0],[1],[1]]) - Ori_ccs
    c2w = np.array(
        [cir_pose_i_ccs.reshape(-1),
         cir_pose_j_ccs.reshape(-1),
         cir_pose_k_ccs.reshape(-1)]
    )
    Cam_wcs = (c2w @ -Ori_ccs)
    if Ori_ccs[2,0] < 0:
        # Fake pose correction
        R_fix = np.array([[ cos(pi), sin(pi), 0],
                          [-sin(pi), cos(pi), 0],
                          [ 0,             0, 1]])
        t_fix = R_fix @ Cam_wcs
        P_fix = np.array([[ cos(pi), sin(pi), 0, -2*t_fix[0,0]],
                          [-sin(pi), cos(pi), 0, -2*t_fix[1,0]],
                          [ 0,             0, 1, +2*t_fix[2,0]],
                          [ 0,             0, 0,             1]])
        # print((P_fix @ [[0],[0],[0],[1]]))
        # print((P_fix @ [[1],[0],[0],[1]]))
        P = (np.concatenate((P,np.array([[0,0,0,1]])),axis=0)@P_fix)[:3,:]
        Ori_ccs = (P @ [[0],[0],[0],[1]])
        cir_pose_i_ccs = (P @ [[1],[0],[0],[1]]) - Ori_ccs
        cir_pose_j_ccs = (P @ [[0],[1],[0],[1]]) - Ori_ccs
        cir_pose_k_ccs = (P @ [[0],[0],[1],[1]]) - Ori_ccs
        c2w = np.array(
            [cir_pose_i_ccs.reshape(-1),
             cir_pose_j_ccs.reshape(-1),
             cir_pose_k_ccs.reshape(-1)]
        )
        Cam_wcs = (c2w @ -Ori_ccs)

    print('Cam_wcs:', Cam_wcs)

    print('c2w:')
    print(c2w)

    print('wcsO in ccs:',Ori_ccs.reshape(3))
    print('wcsX in ccs:',cir_pose_i_ccs.reshape(3))
    print('wcsY in ccs:',cir_pose_j_ccs.reshape(3))
    print('wcsZ in ccs:',cir_pose_k_ccs.reshape(3))
