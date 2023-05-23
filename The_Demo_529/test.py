import json, csv
import argparse
import cv2
import numpy as np
import os, sys
import time, math
import torch
import torchvision.models as models
import dataloader3
import matplotlib.pyplot as plt
import itertools
import scipy.ndimage
from torch.utils.data import TensorDataset, DataLoader
from TrackNet3 import TrackNet3
from dataloader3 import TrackNetLoader
from sklearn.metrics import confusion_matrix
from PIL import Image
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

import pandas as pd
from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from pseudo3d.pseudo3D import Pseudo3d
from pseudo3d.generator import startGL, toRad, toDeg, drawCourt, drawNet

BATCH_SIZE=1
HEIGHT=288
WIDTH=512
the_time = time.time()

parser = argparse.ArgumentParser(description = 'Pytorch TrackNet6')
parser.add_argument('--load_weight', type = str, default='28games_30.tar',help = 'input model weight for predict')
parser.add_argument('--optimizer', type = str, default = 'Ada', help = 'Ada or SGD (default: Ada)')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum fator (default: 0.9)')
parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'weight decay (default: 5e-4)')
parser.add_argument('--seed', type=int, default = 1, help = 'random seed (default: 1)')
parser.add_argument('--lr', type=int, default = 1, help = 'lr')

parser.add_argument("--track", type=str, help='csv file output from TrackNetV2')
parser.add_argument("--fovy", type=float, default=40, help='fovy of visualize window')
parser.add_argument("--height", type=int, default=1060, help='height of visualize window')
parser.add_argument("--width", type=int, default=1920, help='width of visualize window')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ',torch.cuda.is_available())

def angle(v1, v2):
	dx1 = v1[2] - v1[0]
	dy1 = v1[3] - v1[1]
	dx2 = v2[2] - v2[0]
	dy2 = v2[3] - v2[1]
	angle1 = math.atan2(dy1, dx1)
	angle1 = int(angle1 * 180/math.pi)
	angle2 = math.atan2(dy2, dx2)
	angle2 = int(angle2 * 180/math.pi)
	if angle1*angle2 >= 0:
		included_angle = abs(angle1-angle2)
	else:
		included_angle = abs(angle1) + abs(angle2)
		if included_angle > 180:
			included_angle = 360 - included_angle
	return included_angle

def get_point_line_distance(point, line):
	point_x = point[0]
	point_y = point[1]
	line_s_x = line[0]
	line_s_y = line[1]
	line_e_x = line[2]
	line_e_y = line[3]
	if line_e_x - line_s_x == 0:
		return math.fabs(point_x - line_s_x)
	if line_e_y - line_s_y == 0:
		return math.fabs(point_y - line_s_y)
	#斜率
	k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
	#截距
	b = line_s_y - k * line_s_x
	#带入公式得到距离dis
	dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
	return dis

def WBCE(y_pred, y_true):
	eps = 1e-7
	loss = (-1)*(torch.square(1 - y_pred) * y_true * torch.log(torch.clamp(y_pred, eps, 1)) + torch.square(y_pred) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, eps, 1)))
	return torch.mean(loss)

def custom_time(time):
	remain = int(time / 1000)
	ms = (time / 1000) - remain
	s = remain % 60
	s += ms
	remain = int(remain / 60)
	m = remain % 60
	remain = int(remain / 60)
	h = remain
	#Generate custom time string
	cts = ''
	if len(str(h)) >= 2:
		cts += str(h)
	else:
		for i in range(2 - len(str(h))):
			cts += '0'
		cts += str(h)
	
	cts += ':'

	if len(str(m)) >= 2:
		cts += str(m)
	else:
		for i in range(2 - len(str(m))):
			cts += '0'
		cts += str(m)

	cts += ':'

	if len(str(int(s))) == 1:
		cts += '0'
	cts += str(s)

	return cts

def reshape(w, h):
	global tf
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(tf.fovy, w / h, 0.1, 100000.0)
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()

def keyboardFunc(c, x, y):
	global tf, now, track3D
	if ord(c.decode('utf-8')) == 27:
		print('exit...')
		os._exit(0)
	elif ord(c.decode('utf-8')) == ord('d') or ord(c.decode('utf-8')) == ord('D'):
		tf.rad += toRad(5)
		tf.rad %= 2*math.pi
		glutPostRedisplay()
	elif ord(c.decode('utf-8')) == ord('a') or ord(c.decode('utf-8')) == ord('A'):
		tf.rad -= toRad(5)
		tf.rad %= 2*math.pi
		glutPostRedisplay()
	elif ord(c.decode('utf-8')) == ord('w') or ord(c.decode('utf-8')) == ord('w'):
		tf._f += 100
		glutPostRedisplay()
	elif ord(c.decode('utf-8')) == ord('s') or ord(c.decode('utf-8')) == ord('S'):
		tf._f -= 100
		glutPostRedisplay()
	elif ord(c.decode('utf-8')) == ord('n') or ord(c.decode('utf-8')) == ord('N'):
		now = (now + 1) % len(track3D)
		glutPostRedisplay()
	# elif ord(c.decode('utf-8')) == ord(' '):
	#     tf.gt = not tf.gt
	#     glutPostRedisplay()

def sphere(x, y, z, color, size=0.05):
	global tf
	glColor3f(color[0], color[1], color[2])
	glTranslatef(x, y, z)
	gluSphere(tf.quadric, size, 32, 32)
	glTranslatef(-x, -y, -z)

def drawFunc():
	global tf, track3D, now
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	_eye, _obj, _up = tf.rotate(tf.rad)
	gluLookAt(_eye[0], _eye[1], _eye[2],  
			  _obj[0], _obj[1], _obj[2],  
			  _up[0], _up[1], _up[2])

	# Draw badminton court
	drawCourt()
	drawNet()

	# Draw Pseudo3D track
	pred = track3D[now:now+5]
	for i in pred:
		size = 0.1 if tf._f!=0 else 0.1
		if tf.gt:
			sphere(i[0], i[1], i[2], color=[0,1,1], size = size)
		else:
			sphere(i[0], i[1], i[2], color=[1,0,0], size = size)

	# Draw ancher point
	# sphere(tf.td.start_wcs[0], tf.td.start_wcs[1], 0, color=[0,0,1], size = size)
	# sphere(tf.td.end_wcs[0], tf.td.end_wcs[1], 0, color=[0,0,1], size = size)

	print("Focal length offset:", tf._f, "Deg:", toDeg(tf.rad))
	print()

	glutSwapBuffers()



savePath = './predict/'

data = []

valPath = '../data/part1/val/'
valFolders = os.listdir(valPath)
valFolders = sorted(valFolders)

valVideoPath = []
valVideos = []
for folder in valFolders:
	fileName = os.listdir(valPath + folder + '/')
	for fn in fileName:
		if fn[-1] == '4':
			videoPath = valPath + folder + '/' + fn
			valVideoPath.append(videoPath)
			valVideos.append(fn)
ct = 0
predictList = []
data = []

for vp in valVideoPath:
	vn = vp.split('/')[-1]
	cap = cv2.VideoCapture(vp)
	f = open(savePath + vn.split('.')[0] + '.csv', 'w')
	f.write('Frame,Visibility,X,Y,Time\n')

	try:
		total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	except:
		total_frames = -1
	fps = cap.get(cv2.CAP_PROP_FPS)
	ret, frame = cap.read()
	if ret:
		ratio_h = frame.shape[0] / HEIGHT
		ratio_w = frame.shape[1] / WIDTH
		size = (frame.shape[1], frame.shape[0])
	else:
		print("open wabcam error")
		os._exit(0)

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	output_video_path = savePath + vn
	out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame.shape[1],frame.shape[0]))

	print('Beginning predicting......')
	queue_ball=[]
	model = TrackNet3()
	model.to(device)
	if args.optimizer == 'Ada':
		optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)
		#optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
	else:
		optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum = args.momentum)
	checkpoint = torch.load(args.load_weight)
	model.load_state_dict(checkpoint['state_dict'])
	epoch = checkpoint['epoch']
	model.eval()
	count = 0
	count2 = -3
	time_list=[]
	start1 = time.time()
	while True:
		rets = []
		images = []
		frame_times = []
		for idx in range(3):
			# Read frame from wabcam
			ret, frame = cap.read()
			t = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
			rets.append(ret)
			images.append(frame)
			frame_times.append(t)
			count += 1
			count2 += 1

		grays=[]
		if all(rets):
			for img in images:
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				grays.append(img[:,:,0])
				grays.append(img[:,:,1])
				grays.append(img[:,:,2])
			#grays = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
		elif count >= count:
			break
		else:
			print("read frame error. skip...")
			continue

		# TackNet prediction
		unit = np.stack(grays, axis=2)
		unit = cv2.resize(unit, (WIDTH, HEIGHT))
		unit = np.moveaxis(unit, -1, 0).astype('float32')/255
		#unit = np.asarray([unit])
		unit = torch.from_numpy(np.asarray([unit])).to(device)
		with torch.no_grad():
			#start = time.time()
			h_pred = model(unit)
			#end = time.time()
			#time_list.append(end - start)
		h_pred = h_pred > 0.5
		h_pred = h_pred.cpu().numpy()
		h_pred = h_pred.astype('uint8')
		h_pred = h_pred[0]*255

		for idx_f, (image, frame_time) in enumerate(zip(images, frame_times)):
			show = np.copy(image)
			show = cv2.resize(show, (frame.shape[1], frame.shape[0]))
			# Ball tracking
			if np.amax(h_pred[idx_f]) <= 0: # no ball
				f.write(str(count2 + (idx_f))+',0,0,0,'+frame_time+'\n')
				queue_ball.insert(0,None)
				#out.write(image)
			else:
				(cnts, _) = cv2.findContours(h_pred[idx_f].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				rects = [cv2.boundingRect(ctr) for ctr in cnts]
				max_area_idx = 0
				max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
				for i in range(len(rects)):
					area = rects[i][2] * rects[i][3]
					if area > max_area:
						max_area_idx = i
						max_area = area
				target = rects[max_area_idx]
				(cx_pred, cy_pred) = (int(ratio_w*(target[0] + target[2] / 2)), int(ratio_h*(target[1] + target[3] / 2)))
				f.write(str(count2 + (idx_f))+',1,'+str(cx_pred)+','+str(cy_pred)+','+frame_time+'\n')
				cv2.circle(image, (cx_pred, cy_pred), 5, (0,0,255), -1)
				queue_ball.insert(0, (cx_pred, cy_pred))
				#out.write(image)
			for t in range(3):
				try:
					if queue_ball[t] != None:
						#print('hello')
						show = cv2.circle(show, queue_ball[t], 8-t, (175,112,224), -1)
						#out.write(show)
					#if len(queue_players[t]) > 0:
					#	for p in queue_players[t]:
					#		color = (208,133,33) if p[1] > 335 else (151,57,224)
					#		imgg = cv2.circle(i_court, p, 12-t*2, color, -1)
					#		out.write(imgg)
				except:
					break
			out.write(show)
			cv2.imshow('Camera',show)

			try:
				queue_ball.pop(3)
			except:
				pass
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cv2.destroyAllWindows()
	print('Done......')
	f.close()


	list1=[]
	frames=[]
	realx=[]
	realy=[]
	points=[]
	vis=[]
	with open(savePath + vn.split('.')[0] + '.csv', newline='') as csvFile:
		rows = csv.reader(csvFile, delimiter=',')
		num = 0
		count=0
		for row in rows:
			list1.append(row)
		front_zeros=np.zeros(len(list1))
		for i in range(1,len(list1)):
			frames.append(int(float(list1[i][0])))
			vis.append(int(float(list1[i][1])))
			realx.append(int(float(list1[i][2])))
			realy.append(int(float(list1[i][3])))
			if int(float(list1[i][2])) != 0:
				front_zeros[num] = count
				points.append((int(float(list1[i][2])),int(float(list1[i][3])),int(float(list1[i][0]))))
				num += 1
			else:
				count += 1


	points = np.array(points)
	x, y, z = points.T

	Predict_hit_points = np.zeros(len(frames))
	peaks, properties = find_peaks(y, prominence=10)#distance=10) #y是吃非0,所以index有跳

	print('Predict points : ')
	plt.figure(ct)
	plt.plot(z,y*-1,'-')
	predict_hit=[]
	for i in range(len(peaks)):
		print(peaks[i]+int(front_zeros[peaks[i]]))
		predict_hit.append(peaks[i]+int(front_zeros[peaks[i]]))
		#if(peaks[i]+int(front_zeros[peaks[i]]) >= start_point and peaks[i]+int(front_zeros[peaks[i]]) <= end_point):
		#Predict_hit_points[peaks[i]+int(front_zeros[peaks[i]])] = 1

	for i in range(len(peaks)-1):
		start = peaks[i]
		end = peaks[i+1]+1
		plt.plot(z[start:end],y[start:end]*-1,'-')

	print(predict_hit)
	output_name = savePath + vn.split('.')[0] + '_predict_shot.csv'
	with open(output_name,'w', newline='') as csvfile1:
		h = csv.writer(csvfile1)
		h.writerow(['Frame','Visibility','X','Y','Hit'])
		for i in range(len(frames)):
			if i in predict_hit:
				h.writerow([frames[i], vis[i], realx[i], realy[i], 1])
			else:
				h.writerow([frames[i], vis[i], realx[i], realy[i], 0])

	out.release()
	plt.savefig(savePath + vn.split('.')[0] + '.jpg')
	#plt.show()
	ct+=1

	
	ss = 1
	for ie in range(len(predict_hit)):
		data.append([vn, ss, predict_hit[ie], 'X', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'X'])
		ss+=1



# write csv file
header = ['VideoName', 'ShotSeq', 'HitFrame', 'Hitter', 'RoundHead', 'Backhand', 'BallHeight', 'LandingX', 'LandingY', 'HitterLocationX', 'HitterLocationY', 'DefenderLocationX', 'DefenderLocationY', 'BallType', 'Winner']

with open('test.csv', 'w', encoding='UTF8', newline='') as f:
	writer = csv.writer(f)
	writer.writerow(header)
	writer.writerows(data)



"""
# ============================================================================================================ #
#                                                   Pseudo3D                                                   #
# ============================================================================================================ #
parser = argparse.ArgumentParser()
args = parser.parse_args()

# Prepare TrcakNetV2 result
trackdf = pd.read_csv(output_name)
hit_idx = trackdf[trackdf['Hit']>0].reset_index()
print(trackdf)
print(hit_idx)


# Prepare Homography matrix (image(pixel) -> court(meter))
f = open('Hmtx_o2.json')
Hmtx = np.array(json.load(f))
f.close()
print(Hmtx)

# Prepare Intrinsic matix of video
f = open('logitech.json')
Kmtx = np.array(json.load(f)['Kmtx'])
f.close()
print(Kmtx)

track3D = []
now = 0
for p in range(len(hit_idx)-1):
	start = hit_idx.loc[[p]]['Frame'].to_numpy()[0]
	end = hit_idx.loc[[p+1]]['Frame'].to_numpy()[0]
	print(start, end)

	track2D = trackdf[start:end+1].to_numpy()[:,2:4]
	print(track2D)

	# Pseudo3D trajectory transform (2D->3D)
	# tf = Pseudo3d(start_wcs=track2D[0] - [0, 10],
	#     end_wcs=track2D[-1] - [0, 10],
	tf = Pseudo3d(start_wcs=yolo_pred,
		end_wcs=yolo_pred,
		track2D=track2D,
		args=args,
		H=Hmtx,
		K=Kmtx
	)
	track3D = track3D + [i for i in tf.updateF(silence=0)]

# OpenGL visualizer Init
startGL(args)
glutReshapeFunc(reshape)
glutKeyboardFunc(keyboardFunc)
glutDisplayFunc(drawFunc)
glutMainLoop()
"""
