import cv2
import os
def readframe():
    # 读取视频文件
    video_caputre = cv2.VideoCapture("8.gif")
 
    # 获取视频流的参数
    fps = video_caputre.get(cv2.CAP_PROP_FPS)  # 帧率
    width = video_caputre.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    height = video_caputre.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度
    all_frames = video_caputre.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
 
    print("fps:", fps, "\n", "width:", width, "\n", "height:", height, "\n", "all_frames:", all_frames)
 
    # 读取视频流（返回参数：是否读到视频流、图像帧）
    whether, frame = video_caputre.read()
 
    i = 0
    for _ in range(int(all_frames)):
        # cv2.imshow("image", frame)
        # cv2.waitKey(0)
        i += 1
        if i>1100:
            continue
        # 取余数（每10帧保存一张图像）
        if i % 5 == 0:
            cv2.imwrite("./frame2/d{}.png".format(i), frame)
 
        # 读取下一帧图像
        whether, frame = video_caputre.read()
 
    # 释放视频流
    video_caputre.re
    lease()
 
    print("video to images done!")

def rename():
    imglist = os.listdir("./frame/")
    for i,fname in enumerate(imglist):
        if ".png" in fname:
            img = cv2.imread(os.path.join("./frame",fname))
            cv2.imwrite("./frame/{}.png".format(i),img)


#readframe()
rename()