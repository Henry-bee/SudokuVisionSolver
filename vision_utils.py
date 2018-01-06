import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess(image,  default=19, dilate=True):
    '''
    Darken grid and preliminary thresholding
    '''
    img = cv2.GaussianBlur(image,(5,5),0)
    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,default,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    res = cv2.dilate(thresh, kernel) if dilate else thresh
    return res

def orderPoints(cnt):
    '''
    Rearrange 4 point contour in clockwise direction starting from top left
    '''
    cnt = cnt.reshape((4,2))
    rect = np.zeros((4,2), dtype='float32')
    s = cnt.sum(axis=1)
    rect[0] = cnt[np.argmin(s)] #Top-Left
    rect[2] = cnt[np.argmax(s)]
    diff = np.diff(cnt, axis=1)
    rect[1] = cnt[np.argmin(diff)] # Bottom-Left
    rect[3] = cnt[np.argmax(diff)]
    return rect

def four_point_transform(img, contour):
    '''
    Unwarp Puzzle and Enlarge Puzzle
    '''
    
    epsilon = 0.1*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)

    (tl,tr,bl,br)= rect = orderPoints(approx)
    
    # c^2 = x^2 + y^2
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped


def centralize(output, centroid, label):
    img2 = np.zeros(output.shape)
    img2[output == label] = 255.0

    rows, cols = img2.shape 
    target_center = (int(rows/2), int(cols/2))
    shift_x, shift_y = np.round(target_center - centroid).astype('float32')

    M = np.array([[1,0, shift_x], [0,1,shift_y]])

    dst = cv2.warpAffine(img2, M, (cols,rows))
    return dst
   

def largestnConnectedComponents(cell, n=4):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cell, connectivity=4)
    sizes = stats[:, -1]
    indices = np.argsort(sizes)[::-1][:n]
    return output, indices, sizes[indices], centroids[indices]

def digitExists(cell, threshold=0.056):
    total_cell = cell.shape[0] * cell.shape[1]
    if np.count_nonzero(cell) / total_cell > threshold:
        return True
    else:
        return False

def extractdigits(cell_y,cell_x,grid, n=2, this=False):
    '''
    Returns a few images containing the largest components detected within the cell
    If None, returns empty list
    '''
    # x, y coordinates of the cell passed
    y_skips = int(np.floor(grid.shape[0] / 9))
    x_skips = int(np.floor(grid.shape[1] /9 ))
    
    y = y_skips * cell_y
    x = x_skips * cell_x
    
    cell = grid[y:y+y_skips, x:x+x_skips]


    # Check cell white percentage
    if not digitExists(cell):
        return []
    
    # Get the n largest connected Components
    output, indices, sizes, centroids = largestnConnectedComponents(cell, n)
    
    images = []
    centroids = centroids[1:]
    indices = indices[1:]
    for label, centroid in zip(indices, centroids):

        img = centralize(output, centroid, label)
        images.append(img)
        
    return images

def whitePct(img):
    return  np.count_nonzero(img) / (img.shape[0] * img.shape[1]) *100

def aspectratio(cell, highest=0.98, lowest=0.45):
    '''
    Determine that the largest contour has aspect ratio within the given range
    '''
    cell = cell.astype('uint8')
    _, cnt, hiers = cv2.findContours(cell,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[0]
    x,y,w,h = cv2.boundingRect(cnt)

    return w/h < highest and w/h >lowest

def repetitiveThreshold(img, target_pct):
    '''
    Repeatedly tries different blocksize for adaptive thresholding until target pct is achieved
    '''
    thres = 55
    whites = whitePct(img)

    while whites > target_pct:

        threshed = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,thres,2)
        thres += 10
        whites = whitePct(threshed)

    return threshed


def floodfillGrid(img, overlapped, coords, i):
    '''
    Floodfill ith index of coords array, if failed, try i+1 index. Purpose is to remove gridlines
    '''
    coord = coords[i]
    h, w = img.shape
    u_mask = np.zeros((h+2, w+2), np.uint8)
    if overlapped[coord[0]][coord[1]]!= 0:
        try:
            cv2.floodFill(img, u_mask, (coord[0],coord[1]), 0)
        except Exception as e:

            return

def repetitiveFloodfill(dilated, overlapped, coords, i):
    whites = whitePct(dilated)

    while whites > 10:
        floodfillGrid(dilated, overlapped, coords, i)
        i += 1
        whites = whitePct(dilated)

    
def view(img, _size=(800,800)):
    view = cv2.resize(img, _size)
    cv2.imshow('Sudoku', view)
    cv2.waitKey()