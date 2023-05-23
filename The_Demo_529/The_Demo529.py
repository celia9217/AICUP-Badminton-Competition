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
from denoise import smooth

from os.path import isfile, join
import datetime
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

parser = argparse.ArgumentParser(description = 'Pytorch TrackNet6')
parser.add_argument('--load_weight', type = str, default='4camera_50.tar',help = 'input model weight for predict')
parser.add_argument('--optimizer', type = str, default = 'Ada', help = 'Ada or SGD (default: Ada)')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum fator (default: 0.9)')
parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'weight decay (default: 5e-4)')
parser.add_argument('--seed', type=int, default = 1, help = 'random seed (default: 1)')
parser.add_argument('--lr', type=int, default = 1, help = 'lr')

parser.add_argument("--track", type=str, help='csv file output from TrackNetV2')
parser.add_argument("--fovy", type=float, default=40, help='fovy of visualize window')
parser.add_argument("--height", type=int, default=1060, help='height of visualize window')
parser.add_argument("--width", type=int, default=1920, help='width of visualize window')

parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny.cfg', help='*.cfg path')
parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
parser.add_argument('--weights', type=str, default='weights/yolov3-tiny.pt', help='weights path')
parser.add_argument('--source', type=str, default='0', help='source')  # input file/folder, 0 for webcam
parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ',torch.cuda.is_available())

def interp(point):
	for i in range(1, len(point)-2):
		if(point[i][0] == 0 and point[i][1] == 0):
			if(point[i-1][0] != 0 and point[i+1][0] != 0) and (point[i-1][1] != 0 and point[i+1][1] != 0):
				point[i][0] = (point[i-1][0] + point[i+1][0]) // 2
				point[i][1] = (point[i-1][1] + point[i+1][1]) // 2
			elif(point[i-1][0] != 0 and point[i+2][0] != 0) and (point[i-1][1] != 0 and point[i+2][1] != 0):
				point[i][0] = point[i-1][0] + (point[i+2][0] - point[i-1][0]) // 3
				point[i][1] = point[i-1][1] + (point[i+2][1] - point[i-1][1]) // 3
				point[i+1][0] = point[i-1][0] + 2 * (point[i+2][0] - point[i-1][0]) // 3
				point[i+1][1] = point[i-1][1] + 2 * (point[i+2][1] - point[i-1][1]) // 3
	return point
	
def getHimg(path):
	s = cv2.FileStorage(path, cv2.FileStorage_READ)
	hmtx = s.getNode('Hmtx').mat()
	s.release()
	img_court = cv2.imread("badminton_court.png")
	return hmtx, img_court

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
	pred = track3D[now]
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
	
#########################################

BATCH_SIZE=1
TRACKNET_HEIGHT=288
TRACKNET_WIDTH=512
YOLO_HEIGHT=384
YOLO_WIDTH=512
the_time = time.time()
f = open('output/' + str(the_time)+'_predict.csv', 'w')
f.write('Frame,Visibility,X,Y,Time\n')

################# video #################
cap = cv2.VideoCapture('demo4.mp4')
try:
	total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
except:
	total_frames = -1
fps = cap.get(cv2.CAP_PROP_FPS)
ret, frame = cap.read()
if not ret:
	print("open wabcam error")
	os._exit(0)
VIDEO_HEIGHT=frame.shape[0]
VIDEO_WIDTH=frame.shape[1]
RATIO_TRACKNET2VIDEO_H = VIDEO_HEIGHT / TRACKNET_HEIGHT
RATIO_TRACKNET2VIDEO_W = VIDEO_WIDTH / TRACKNET_WIDTH
RATIO_VIDEO2YOLO_H = YOLO_HEIGHT / VIDEO_HEIGHT
RATIO_VIDEO2YOLO_W = YOLO_WIDTH / VIDEO_WIDTH
RATIO_YOLO2VIDEO_H = VIDEO_HEIGHT / YOLO_HEIGHT
RATIO_YOLO2VIDEO_W = VIDEO_WIDTH / YOLO_WIDTH

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path =  'output/' + str(the_time)+'_predict.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (YOLO_WIDTH,YOLO_HEIGHT))
################# video #################


print('Beginning predicting......')

################# TrackNetV2 #################
model = TrackNet3()
model.to(device)
if args.optimizer == 'Ada':
	optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)
else:
	optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum = args.momentum)
checkpoint = torch.load(args.load_weight)
model.load_state_dict(checkpoint['state_dict'])
epoch = checkpoint['epoch']
model.eval()
current = 0
count2 = -3
time_list=[]
start1 = time.time()
################# TrackNetV2 #################



################# Yolov3 #################
yolo = Darknet(args.cfg, args.img_size)
yolo.to(device).eval()
if args.weights.endswith('.pt'):
	yolo.load_state_dict(torch.load(args.weights, map_location=device)['model']) # pytorch format
else:
	load_darknet_weights(yolo, args.weights) # darknet format
################# Yolov3 #################



################# Visualization #################
hmtx, img_court = getHimg("mtx/Hmtx.yml")
CCS_SCALE = 2
img_court = cv2.resize(img_court, (img_court.shape[1]//CCS_SCALE, img_court.shape[0]//CCS_SCALE))
names = load_classes(args.names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
#out_c = cv2.VideoWriter('{} Player Movement.mp4'.format(str(the_time)), fourcc, fps, (img_court.shape[1], img_court.shape[0]))

queue_ball = [] # list of position of 1 ball
queue_players = [] # list of position of n people
top_player = [] # list of position of top players
bot_player = [] # list of position of bot players
top_player_court = []
bot_player_court = []
top_box_lefttop = []
top_box_rightbot = []
bot_box_lefttop = []
bot_box_rightbot = []
################# Visualization #################



while True:
	# Prepare images
	rets = []
	images = []
	frame_times = []
	for idx in range(3):
		ret, frame = cap.read()
		t = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
		rets.append(ret)
		images.append(frame)
		frame_times.append(t)
		count2 += 1
		current += 1

	grays=[]
	if all(rets):
		for img in images:
			grays.append(img[:,:,2])
			grays.append(img[:,:,1])
			grays.append(img[:,:,0])
	elif current >= total_frames:
		print("video end. break")
		break
	else:
		print("read frame error. skip...")
		continue

	# TackNetV2 prediction
	unit = np.stack(grays, axis=2)
	unit = cv2.resize(unit, (TRACKNET_WIDTH, TRACKNET_HEIGHT))
	unit = np.moveaxis(unit, -1, 0).astype('float32')/255
	unit = torch.from_numpy(np.asarray([unit])).to(device)
	with torch.no_grad():
		h_pred = model(unit)
		
	h_pred = h_pred > 0.5
	h_pred = h_pred.cpu().numpy()
	h_pred = h_pred.astype('uint8')
	h_pred = h_pred[0]*255

	# Yolov3 prediction
	unit_yolo = [cv2.resize(img, (YOLO_WIDTH,YOLO_HEIGHT)) for img in images]
	unit_yolo = np.moveaxis(np.asarray(unit_yolo), -1, 1)
	unit_yolo = torch.from_numpy(unit_yolo).to(device).float()  # uint8 to fp16/32
	unit_yolo /= 255.0  # 0 - 255 to 0.0 - 1.0
	b_pred = yolo(unit_yolo, augment=args.augment)[0]

	# Deal with Prediction
	for idx_f, (image, frame_time) in enumerate(zip(images, frame_times)): # for each 1 image
		show_court = np.copy(img_court)
		show = cv2.resize(np.copy(image), (YOLO_WIDTH,YOLO_HEIGHT))
		
		# Ball Tracking
		if np.amax(h_pred[idx_f]) <= 0: # no ball detected in current frame
			f.write(str(count2 + (idx_f))+',0,0,0,'+frame_time+'\n')
			queue_ball.insert(0,None)
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
			(cx_pred, cy_pred) = (int(RATIO_TRACKNET2VIDEO_W*(target[0] + target[2] / 2)), int(RATIO_TRACKNET2VIDEO_H*(target[1] + target[3] / 2)))
			f.write(str(count2 + (idx_f))+',1,'+str(cx_pred)+','+str(cy_pred)+','+frame_time+'\n')
			# cv2.circle(image, (cx_pred, cy_pred), 5, (0,0,255), -1) # draw current frame
			queue_ball.insert(0, (((cx_pred*YOLO_WIDTH)//VIDEO_WIDTH, (cy_pred*YOLO_HEIGHT)//VIDEO_HEIGHT)))
			#queue_ball.insert(0, [(int(cx_pred*RATIO_VIDEO2YOLO_W), int(cy_pred*RATIO_VIDEO2YOLO_H))])
			
		
		#Player Movement Tracking
		det = non_max_suppression(b_pred[np.newaxis,idx_f], args.conf_thres, args.iou_thres,
								   multi_label=False, classes=args.classes, agnostic=args.agnostic_nms)[0] # Apply NMS with batch size 1
		tmp_players = []
		bot_count = 0
		top_count = 0
		if det is not None and len(det):
			# Rescale boxes from imgsz to image_y size
			det[:, :4] = scale_coords(unit_yolo.shape[2:], det[:, :4], (YOLO_HEIGHT, YOLO_WIDTH, 3)).round()

			# Print results
			for c in det[:, -1].detach().unique():
				n = (det[:, -1] == c).sum()  # detections per class
			
			for *xyxy, conf, cls in reversed(det):
				if int(cls) != 0: # not human
					continue
				_xyxy = [i.to('cpu').detach().numpy() for i in xyxy]
				x_i, y_i = ((_xyxy[0]+_xyxy[2])/2, _xyxy[3]) # bottom center of bbox in WCS
				p_c = (hmtx @ np.array([[x_i], [y_i], [1]])).reshape(-1)
				x_c, y_c = (int(p_c[0]/p_c[2])//CCS_SCALE, int(p_c[1]/p_c[2])//CCS_SCALE) # bottom center of bbox in CCS (cm)
				
				is_bot_player = y_c > (1340/2)//CCS_SCALE
				if is_bot_player and bot_count == 0: # only got 1 bot player in current frame
					bot_player.append([int(x_i*RATIO_YOLO2VIDEO_W), int(y_i*RATIO_YOLO2VIDEO_H)])
					bot_player_court.append([int(x_c), int(y_c)])
					bot_box_lefttop.append([int(_xyxy[0]),int(_xyxy[1])])
					bot_box_rightbot.append([int(_xyxy[2]),int(_xyxy[3])])
					bot_count+=1
					
				elif not is_bot_player and top_count == 0: # only got 1 top player in current frame
					top_player.append([int(x_i*RATIO_YOLO2VIDEO_W), int(y_i*RATIO_YOLO2VIDEO_H)])
					top_player_court.append([int(x_c), int(y_c)])
					top_box_lefttop.append([int(_xyxy[0]),int(_xyxy[1])])
					top_box_rightbot.append([int(_xyxy[2]),int(_xyxy[3])])
					top_count+=1
					
				color = (208,133,33) if y_c > (1340/2)//CCS_SCALE else (151,57,224)
				cv2.circle(show, (int(x_i), int(y_i)), 5, color, -1)
				tmp_players.append((x_c, y_c))
	
		# no player detected in current frame
		if top_count == 0:
			top_player.append([0,0])
			top_player_court.append([0,0])
			top_box_lefttop.append([0,0])
			top_box_rightbot.append([0,0])
		if bot_count == 0:
			bot_player.append([0,0])
			bot_player_court.append([0,0])
			bot_box_lefttop.append([0,0])
			bot_box_rightbot.append([0,0])
		queue_players.insert(0, tmp_players)
		
		for t in range(3):
			try:
				# Draw ball
				if queue_ball[t] != None:
					cv2.circle(show, queue_ball[t], 8-t, (0,0,255), -1)
				# Draw player
				if len(queue_players[t]) > 0:
					for p in queue_players[t]:
						color = (208,133,33) if p[1] > (1340/2)//CCS_SCALE else (151,57,224)
						cv2.circle(show_court, p, 12-t*2, color, -1)
			except:
				break
		out.write(show)
		cv2.imshow('Player Movement', show_court)
		cv2.imshow('Camera', show)

		try:
			queue_ball.pop(-1)
			queue_players.pop(-1)
		except:
			print('........')
			exit()
		
	if cv2.waitKey(10) & 0xFF == 27:
		break

top_player = np.array(top_player)
bot_player = np.array(bot_player)
cv2.destroyAllWindows()
print('Done......')
out.release()
f.close()

try:
	smooth('output/'+str(the_time)+'_predict.csv')
	TrackNet_result = 'output/'+str(the_time)+'_predict_denoise.csv'
except:
	TrackNet_result = 'output/'+str(the_time)+'_predict.csv'


list1=[]
frames=[]
realx=[]
realy=[]
points=[]
vis=[]
with open(TrackNet_result, newline='') as csvFile:
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
peaks, properties = find_peaks(y, prominence=10) #y是吃非0,所以index有跳

start_point = 0
predict_hit=[]
for i in range(len(y)-1):
    if((y[i] - y[i+1]) / (z[i+1] - z[i]) >= 4):
        start_point = i+front_zeros[i]
        predict_hit.append(int(start_point))
        break

print('Predict points : ')
plt.plot(z,y*-1,'-')

for i in range(len(peaks)):
	print(peaks[i]+int(front_zeros[peaks[i]]),end=', ')
	if(peaks[i]+int(front_zeros[peaks[i]]) > start_point):
		predict_hit.append(peaks[i]+int(front_zeros[peaks[i]]))


	
'''
# ============================================================================================================ #
#                                          CSV with Tracknet                                                   #
# ============================================================================================================ #


output_name = 'output/' + str(the_time)+'predict_shot.csv'
with open(output_name,'w', newline='') as csvfile1:
	h = csv.writer(csvfile1)
	h.writerow(['Frame','Visibility','X','Y','Hit'])
	for i in range(len(frames)):
		if i in predict_hit:
			h.writerow([frames[i], vis[i], realx[i], realy[i], 1])
		else:
			h.writerow([frames[i], vis[i], realx[i], realy[i], 0])

	
'''
# ============================================================================================================ #
#                             CSV with Openpose and Tracknet                                                   #
# ============================================================================================================ #

# little smooth
top_player = interp(top_player)
bot_player = interp(bot_player)
top_player_court = interp(top_player_court)
bot_player_court = interp(bot_player_court)
top_box_lefttop = interp(top_box_lefttop)
top_box_rightbot = interp(top_box_rightbot)
bot_box_lefttop = interp(bot_box_lefttop)
bot_box_rightbot = interp(bot_box_rightbot)

output_name = 'output/' + str(the_time)+'_predict_all.csv'
with open(output_name,'w', newline='') as csvfile1:
	h = csv.writer(csvfile1)
	h.writerow(['Frame','Visibility','X','Y','Hit', 'Hit_position_x','Hit_position_y', 'Top_x','Top_y', 'Bot_x', 'Bot_y','Top_box_lefttop', 'Top_box_rightbot', 'Bot_box_lefttop', 'Bot_box_rightbot'])
	for i in range(len(frames)):
		if i in predict_hit:
			if(realx[i] != 0 or realy[i] != 0):
				visibility = 1
			else:
				visibility = 0
			if abs(realx[i] - top_player[i][0]) < abs(realx[i] - bot_player[i][0]):
					h.writerow([frames[i], visibility, realx[i], realy[i], 'Top', top_player[i][0], top_player[i][1], top_player_court[i][0], top_player_court[i][1], bot_player_court[i][0], bot_player_court[i][1],top_box_lefttop[i], top_box_rightbot[i], bot_box_lefttop[i], bot_box_rightbot[i]])
			else:
					h.writerow([frames[i], visibility, realx[i], realy[i], 'Bot', bot_player[i][0], bot_player[i][1], top_player_court[i][0], top_player_court[i][1], bot_player_court[i][0], bot_player_court[i][1],top_box_lefttop[i], top_box_rightbot[i], bot_box_lefttop[i], bot_box_rightbot[i]])
		else:
			h.writerow([frames[i], visibility, realx[i], realy[i], 0, 0, 0, top_player_court[i][0], top_player_court[i][1], bot_player_court[i][0], bot_player_court[i][1],top_box_lefttop[i], top_box_rightbot[i], bot_box_lefttop[i], bot_box_rightbot[i]])



predict_hit.insert(0,30)
for i in range(1,len(predict_hit)):
	if((top_player_court[predict_hit[i-1]][0] != 0 or top_player_court[predict_hit[i-1]][1] != 0) and (top_player_court[predict_hit[i]][0] != 0 or top_player_court[predict_hit[i]][1] != 0)):
		cv2.arrowedLine(img_court, (top_player_court[predict_hit[i-1]][0], top_player_court[predict_hit[i-1]][1]), (top_player_court[predict_hit[i]][0], top_player_court[predict_hit[i]][1]), (175,112,224), 2)
	if((bot_player_court[predict_hit[i-1]][0] != 0 or bot_player_court[predict_hit[i-1]][1] != 0) and (bot_player_court[predict_hit[i]][0] != 0 or bot_player_court[predict_hit[i]][1] != 0)):
		cv2.arrowedLine(img_court, (bot_player_court[predict_hit[i-1]][0], bot_player_court[predict_hit[i-1]][1]), (bot_player_court[predict_hit[i]][0], bot_player_court[predict_hit[i]][1]), (209,164,105), 2)
	if(top_player_court[predict_hit[i]][0] != 0 or top_player_court[predict_hit[i]][1] != 0):
		cv2.circle(img_court, (top_player_court[predict_hit[i]][0], top_player_court[predict_hit[i]][1]), 5, (151,57,224), -1)
	if(bot_player_court[predict_hit[i]][0] != 0 or bot_player_court[predict_hit[i]][1] != 0):
		cv2.circle(img_court, (bot_player_court[predict_hit[i]][0], bot_player_court[predict_hit[i]][1]), 5, (208,133,33), -1)
cv2.imwrite('output/{} Player Movement On Court.png'.format(str(the_time)), img_court)

print('Detect Complete')

# ============================================================================================================ #
#                                                   Pseudo3D                                                   #
# ============================================================================================================ #


import pandas as pd
from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from pseudo3d.pseudo3D import Pseudo3d
from pseudo3d.generator import startGL, toRad, toDeg, drawCourt, drawNet

# Prepare TrcakNetV2 result
trackdf = pd.read_csv(output_name)
hit_idx = trackdf[trackdf['Hit']!='0'].reset_index()
print(trackdf)
print(hit_idx)


# Prepare Homography matrix (image(pixel) -> court(meter))
f = open('Hmtx.json')
Hmtx = np.array(json.load(f))
f.close()
print(Hmtx)

# Prepare Intrinsic matix of video
f = open('Kcam02.json')
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

	start_ics = trackdf.loc[[start]].to_numpy()[:,5:7]
	end_ics = trackdf.loc[[end]].to_numpy()[:,5:7]
	print('s',np.array(start_ics).T)
	print('e',np.array(end_ics).T)

	# Pseudo3D trajectory transform (2D->3D)
	# tf = Pseudo3d(start_wcs=track2D[0] - [0, 10],
	#     end_wcs=track2D[-1] - [0, 10],
	tf = Pseudo3d(start_wcs=np.array(start_ics).reshape(-1),
		end_wcs=np.array(end_ics).reshape(-1),
		track2D=track2D,
		args=args,
		H=Hmtx,
		K=Kmtx
	)
	track3D.append(tf.updateF(silence=0))

# OpenGL visualizer Init
startGL(args)
glutReshapeFunc(reshape)
glutKeyboardFunc(keyboardFunc)
glutDisplayFunc(drawFunc)
glutMainLoop()
