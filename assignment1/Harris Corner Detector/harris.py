import cv2
import numpy as np

def get_gaussian_kernel_2D(ksize, sigma):

    gaussian_kernel = np.zeros((ksize, ksize))
    offset = int(ksize / 2)
    for i in range(-offset, offset + 1):
        for j in range(-offset, offset + 1):
            gaussian_kernel[i + offset][j + offset] = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

    return gaussian_kernel

def mark_corner_points(image,window_size,ksize,k,gaussian_weights=False,std=1):
    """
    Return image with corner points marked and a list of corners
    """
    
    new_image = image.copy()
    new_image = cv2.cvtColor(new_image,cv2.COLOR_GRAY2RGB)
    
    corners = []
    threshold = 8000
    gaussian_kernel = get_gaussian_kernel_2D(window_size,std)
    #Gradient using sobel operator
    sx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize = ksize)
    sy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize = ksize)
    
    Ixx = sx**2
    Iyy = sy**2
    Ixy = sx*sy
    
    
    R_score_list =[]
    rows,cols = image.shape
    
    offset = window_size/2
    
    corner_count = 0
    for i in range(offset,rows-offset):
        for j in range(offset,cols-offset):
            
            xx_window = Ixx[i-offset:i+offset+1,j-offset:j+offset+1]
            yy_window = Iyy[i-offset:i+offset+1,j-offset:j+offset+1]
            xy_window = Ixy[i-offset:i+offset+1,j-offset:j+offset+1]
            
            if gaussian_weights:
                xx_window = xx_window*gaussian_kernel
                yy_window = yy_window*gaussian_kernel
                xy_window = xy_window*gaussian_kernel
            
            sum_xx = np.sum(xx_window)
            sum_yy = np.sum(yy_window)
            sum_xy = np.sum(xy_window)
            
            det_window = sum_xx*sum_yy - sum_xy*sum_xy
            trace_window = sum_xx + sum_yy
            
            R_score = det_window - k*trace_window*trace_window
            
            
            corners.append([i,j,R_score])
    
    corners = np.array(corners)
    threshold = 0.005*np.max(corners[:,2])
    
    for corner in corners:
        if corner[2] > threshold:
            corner_count+=1
            cv2.circle(new_image,(int(corner[1]),int(corner[0])),3,(0,0,255),1)
                
    print corner_count       
    return new_image,corners