import cv2
import numpy as np

DIR = '.\\images\\'
def save_image(img,name,ID):
    cv2.imwrite(DIR+name+ID+'.png',img)
    
def get_angle_direction(angle):
    """
    Converts angle to degrees and classified into 4 directions; vertical, horizontal and two diagonal directions
    
    Parameters
    -------------
    1. angle : float - angle in radians
    
    Return value
    --------------
    Angle in degrees
    
    """
    angle = np.rad2deg(angle)%180
    
    VERTICAL_CUT = 22.5
    DIAGONAL_CUT = 45
    if angle <= VERTICAL_CUT or angle >= 180 - VERTICAL_CUT:
        direction = 0
    elif angle <= VERTICAL_CUT + DIAGONAL_CUT:
        direction = 45
    elif angle <=VERTICAL_CUT + 2*DIAGONAL_CUT:
        direction = 90
    else:
        direction = 135
    
    return direction

def get_gradient(image,ksize):
    """
    Returns gradient intensity and orientation of 'image' using sobel operator of size 'ksize'
    """
    #Sobel operations
    sobelX = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=ksize)
    sobelY = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=ksize)
    
    #Calculating gradient intensity and orientation
    edge_gradient = np.hypot(sobelX,sobelY)
    edge_angle = np.arctan2(sobelY,sobelX)
    
    return edge_gradient,edge_angle

def non_maximum_suppression(image,edge_angle):
    """
    Removes any unwanted pixels which may not constitute the edge. For this, at every pixel, pixel is checked if it is a 
    local maximum in its neighborhood in the direction of gradient.
    
    Parameters
    -------------
    1. image      : ndarray - gradient intensity image
    2. edge_angle : ndarray - gradient orientation image
    
    Return value
    --------------
    Suppressed image
    
    """
    rows,cols = image.shape
    suppression = np.zeros((rows,cols),dtype=np.int32)
    
    for i in range(rows):
        for j in range(cols):
            #Boundary pixels are not considered
            if i==0 or j==0 or i==rows-1 or j==cols -1:
                continue
            
            direction = get_angle_direction(edge_angle[i][j])

            if direction == 0 and image[i][j] >= image[i][j-1] and image[i][j] >= image[i][j+1]:
                suppression[i][j] = image[i][j]
            elif direction ==45 and image[i][j] >= image[i-1][j+1] and image[i][j] >= image[i+1][j-1]:
                suppression[i][j] = image[i][j]
            elif direction == 90 and image[i][j] >= image[i+1][j] and image[i][j] >= image[i-1][j]:
                suppression[i][j] = image[i][j]
            elif direction == 135 and image[i][j] >= image[i-1][j-1] and image[i][j] >= image[i+1][j+1]:
                suppression[i][j] = image[i][j]
                               
    return suppression
    
    
def hysteresis_threshold(image,minVal,maxVal):
    """
    Decides which are all edges are really edges and which are not
    
    Parameters
    -------------
    1. image  : ndarray - suppressed edge image
    2. minVal  : Integer  -  first threshold for the hysteresis procedure
    3. maxVal  : Integer  -  second threshold for the hysteresis procedure
    
    Return value
    -------------------
    Image with only strong edges
    """
    
    WEAK_EDGE = 100
    STRONG_EDGE = 255
    
    if minVal>maxVal:
        raise ValueError("minVal should be less than or equal to maxVal")
    
    image[image > maxVal] = STRONG_EDGE
    image[np.logical_and(image >= minVal, image <= maxVal)] = WEAK_EDGE
    image[image < minVal] = 0
    
    row,cols = image.shape
    for i in range(row):
        for j in range(cols):
            if image[i][j] == WEAK_EDGE:
                try:
                    if STRONG_EDGE in [image[i+1][j],image[i-1][j],image[i][j+1],image[i][j-1],image[i+1][j+1],image[i-1][j-1],image[i+1][j-1],image[i-1][j+1]]:
                        image[i][j] = STRONG_EDGE
                    else:
                        image[i][j] = 0
                except IndexError as error:
                    pass
    return image
    
    
def Canny(image,outname,minVal,maxVal,ksize=3,Gaus_Size=3):
    """
    Finds the edges in an image
    
    Steps involved:
    1. Noise reduction using Gaussian filter
    2. Finding intensity gradient of the image using Sobel kernels
    3. Non-maximum suppression to remove unwanted pixels
    4. Hysteresis thresholding to get the strong edges
    
    Parameters
    --------------
    1. image   : ndarray input - image
    2. minVal  : Integer  -  first threshold for the hysteresis procedure
    3. maxVal  : Integer  -  second threshold for the hysteresis procedure
    4. ksize   : Integer  -  kernel size for the Sobel operator. (Default = 3)
    
    Return value
    --------------
    Binary edge image
    
    """
    
    #Noise reduction using Gaussian filter
    blured_img = cv2.GaussianBlur(image,(Gaus_Size,Gaus_Size),0)
    save_image(blured_img,'blured_img',outname)
    
    #Finding intensity gradient of the image using Sobel kernels
    edge_gradient,edge_angle = get_gradient(image,ksize)
    save_image(edge_gradient,'edge_gradient',outname)
    
    #Non-maximum suppression to remove unwanted pixels
    suppressed_edge_img = non_maximum_suppression(edge_gradient,edge_angle)
    save_image(suppressed_edge_img,'suppressed_edge_img',outname)
    
    #Hysteresis thresholding to get the strong edges
    h_thresh_image = hysteresis_threshold(suppressed_edge_img,minVal,maxVal)
    
    return h_thresh_image
