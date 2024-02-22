import os
import time

import cv2  # pip install opencv-python
import imageio
import numpy as np  # pip install numpy
import PySimpleGUI as sg  # pip install pysimplegui
from dt_apriltags import Detector
from PIL import Image
from ultralytics import YOLO

from util import *

with np.load("../intrinsic_calib.npz") as X:
    mtx, dist = [X[i] for i in ('mtx', 'dist')]
# train

# model = YOLO('yolov8s.yaml').load('yolov8s.pt')
# results = model.train(data='./dataset/tripods.yaml', epochs=10)



# # single infer
# model = YOLO('./ultralytics/yolov8n.pt')

# results = model('./frame/1.png')  

# for r in results:
#     im_array = r.plot()  
#     im = Image.fromarray(im_array[..., ::-1]) 
#     im.show() 
#     im.save('results.jpg') 





# frames = []
# # Load a model
model_tripod = YOLO('../runs/detect/train/weights/best.pt')  
model_basic = YOLO(('../yolov8l.pt') )  
at_detector = Detector(families='tag36h11')

# source = '/home/sc/yongqi/yolo/3.gif'
# results = model(source, stream=True, conf=0.4)
# for r in results:
#     im_array = r.plot()  # 绘制包含预测结果的BGR numpy数组
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
#     frames.append(im)
# imageio.mimsave("demo3.gif", frames, 'GIF', duration=0.1)




capture_root = '../capture'
layout = [
    [sg.Image(filename='', key='raw',size=(500,400))],
    [sg.Text('fps = 0',  key="text")],
    [sg.Image(filename='', key='pred')],
    [sg.Radio('None', 'Radio', True, size=(10, 1))],
    [sg.Checkbox('Tripod', key='tripod')],
    [sg.Checkbox('Calibration', key='calib')],
    [sg.Checkbox('FindTag', key='tag')],
    [sg.Checkbox('Chair (check Tag and uncheck Calib)', key='chair')],
    [sg.Button('Capture', size=(20, 3))],
    [sg.Button('Exit', size=(20, 3))]
]

window = sg.Window('camera',
            layout,
            location=(1000, 500),
            resizable=True,
            element_justification='c',
            font=("Arial Bold",20),
            finalize=True)

cap = cv2.VideoCapture("/dev/v4l/by-id/usb-046d_Logitech_BRIO_13100349-video-index0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)  

while True:
    event, values = window.read(timeout=0, timeout_key='timeout')
    #read from cam
    ret, frame = cap.read()

    if values['calib']:
        #intri calib
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(frame.shape[1],frame.shape[0]),0,(frame.shape[1],frame.shape[0]))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # clip the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
    else:
        dst = frame

    # model prediction
    if values['tripod']:
        pred = model_tripod(dst, conf=0.4, iou=0.1)
    else:
        pred = model_basic(dst)

    marked = pred[0].plot()  
    boxes = pred[0].boxes.xyxy.tolist()
    classes = pred[0].boxes.cls.tolist()
    confidences = pred[0].boxes.conf.tolist()
    fps = 1000/pred[0].speed["inference"]
    window['text'].update("fps = {:.2f}".format(fps))
    gray = cv2.cvtColor(marked, cv2.COLOR_BGR2GRAY)
    fx,fy,cx,cy = mtx[0][0],mtx[1][1],mtx[0][2], mtx[1][2]
    cam_params = [fx, fy, cx, cy]
    tags = at_detector.detect(gray,estimate_tag_pose=True,camera_params=cam_params,tag_size=0.15)

    if len(tags)>0:
        rotm = tags[0].pose_R
        trasm = tags[0].pose_t
        homo = tags[0].homography

    if values['chair']:
        for box, label, conf in zip(boxes, classes, confidences):
            if label == 56:
                if conf>=0.8:
                    center_x = box[0] + (box[2] - box[0]) /2 # col from up left
                    # center_y = box[1] + (box[3] - box[1])*3/4 # row from up left
                    center_y = box[3]
                    cv2.circle(marked, (int(center_x),int(center_y)), 6, (0, 255, 255), 3)
                    point = [[center_x,center_y]]
                    point = np.array(point,dtype='float')
                    dir_cam = pixel2ray(point, mtx, dist).reshape((3,1))
                    origin_w = -np.dot(rotm.T, (np.array([[0],[0],[0]])- trasm))
                    dir_w = np.dot(rotm.T, dir_cam)
                    normal = np.array([[0,0,1]])
                    plane_x0 = np.array([[1,0,0]]).T
                    t = (np.dot(normal, origin_w)-np.dot(normal,plane_x0))/np.dot(normal,dir_w)
                    intersection_w = (origin_w - t * dir_w)
                    intersection_w = np.around(intersection_w,decimals=2)
    
                    text_str = str(intersection_w.tolist())
                    cv2.putText(marked,text_str,(int(center_x),int(center_y)),thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0,0,255))
                    with open('value1.txt', 'w') as file:
                        file.write(str(text_str))
                else:
                    with open('value1.txt', 'w') as file:
                        file.write("no valid chairs")




    
    if values['tag']:

        
        for tag in tags:

            # if tag.families == "":
            #     origin = tag
            rotm = tag.pose_R
            trasm = tag.pose_t
            homo = tag.homography
            cv2.circle(marked, tuple(tag.corners[0].astype(int)), 4, (255, 0, 0), 2) # left-top
            cv2.circle(marked, tuple(tag.corners[1].astype(int)), 4, (255, 0, 0), 2) # right-top

            cv2.circle(marked, tuple(tag.corners[2].astype(int)), 4, (255, 0, 0), 2) # right-bottom
            cv2.circle(marked, tuple(tag.corners[3].astype(int)), 4, (255, 0, 0), 2) # left-bottom

            # intersection_c_list = []

            # for c in tag.corners:
            #     point = c
            #     point = np.array(point,dtype='float')
            #     dir_cam = pixel2ray(point, mtx, dist).reshape((3,1))
            #     origin_w = -np.dot(rotm.T, (np.array([[0],[0],[0]])- trasm))
            #     dir_w = np.dot(rotm.T, dir_cam)
            #     normal = np.array([[0,0,1]])
            #     plane_x0 = np.array([[1,0,0]]).T
            #     t = (np.dot(normal, origin_w)-np.dot(normal,plane_x0))/np.dot(normal,dir_w)
            #     intersection_w = (origin_w - t * dir_w)
            #     intersection_w = np.around(intersection_w,decimals=5)
            #     intersection_c_list.append(intersection_w)

            # scale_x = (abs(2/(intersection_c_list[1][0] - intersection_c_list[0][0])) + abs(2/(intersection_c_list[3][0] - intersection_c_list[2][0])))/2
            # scale_y = (abs(2/(intersection_c_list[3][1] - intersection_c_list[0][1])) + abs(2/(intersection_c_list[2][1] - intersection_c_list[3][1])))/2


            cv2.circle(marked, tuple(tag.center.astype(int)), 4, (255, 0, 0), 4) #标记apriltag码中心点
            # M, e1, e2 = at_detector.detection_pose(tag, cam_params)
 

            P = np.concatenate((rotm, trasm), axis=1) #相机投影矩阵

            P = np.matmul(mtx,P)
            x = np.matmul(P,np.array([[-1],[0],[0],[1]]))  
            x = x / x[2]
            y = np.matmul(P,np.array([[0],[-1],[0],[1]]))
            y = y / y[2]
            z = np.matmul(P,np.array([[0],[0],[-1],[1]]))
            z = z / z[2]
            cv2.line(marked, tuple(tag.center.astype(int)), tuple(x[:2].reshape(-1).astype(int)), (0,0,255), 2) #x轴红色
            cv2.line(marked, tuple(tag.center.astype(int)), tuple(y[:2].reshape(-1).astype(int)), (0,255,0), 2) #y轴绿色
            cv2.line(marked, tuple(tag.center.astype(int)), tuple(z[:2].reshape(-1).astype(int)), (255,0,0), 2) #z轴蓝色



    # GUI update
    rawbytes = cv2.imencode('.png', frame[::2,::2,:])[1].tobytes()
    window['raw'].update(data=rawbytes)
    predbytes = cv2.imencode('.png', marked)[1].tobytes()
    window['pred'].update(data=predbytes)


    if event == 'Capture':
        cur_time = time.localtime()
        imgname = time.strftime("%H-%M%S.png",cur_time)
        cv2.imwrite(os.path.join(capture_root,imgname),marked)
    if event == 'Exit':
        break


    

cap.release()
window.close()
