from dt_apriltags import Detector
import numpy
import os
import cv2

img_dir = "Picture1.png"
img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)


at_detector = Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

tags = at_detector.detect(img, estimate_tag_pose=False, camera_params=None, tag_size=None)
