import cv2
import numpy as np
import sys
from harris import *

DIR = '.\\output\\'
try:
    image_file_path = sys.argv[1]
except IndexError as error:
    print error
    sys.exit()
    
img = cv2.imread(image_file_path,0)

#Gaussian blur to remove noise
img = cv2.GaussianBlur(img,(3,3),0)

#Increasing contrast using Histogram Equalisation
img = cv2.equalizeHist(img)

window_size = 3
ksize =3
k=0.05

corner_marked_image,corners = mark_corner_points(img,window_size,ksize,k,True)
cv2.imwrite(DIR+'corner_marked_'+image_file_path[:-4]+'.png',corner_marked_image)