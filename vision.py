import os

import numpy as np

import cv2
import tensorflow as tf
from vision_utils import (extractdigits, four_point_transform, preprocess,
                   repetitiveFloodfill, repetitiveThreshold, view, largestnConnectedComponents, centralize, aspectratio)


def puzzle2array(img):
    '''
    Return a 2D array with image array where there is number, and empty arr if none
    '''
    print('Begin extraction of digit image')
    puzzle = []
    for y in range(0,9):
        row =[]
        for x in range(0,9):
            digit = extractdigits(y, x, img)
            
            if len(digit) < 1: 
                row.append([])
            else:
                print('Digit detected at cell %s %s'%(y,x))
                print (digit)
                print ('Aspect Ratio: %s'%(aspectratio(digit[0])))
                print ("")
                if aspectratio(digit[0]):
                    row.append(digit[0])
                else:
                    row.append([])
        puzzle.append(row)

    print('Complete Puzzle2Arr')
    return puzzle


def extractPuzzle(gray):
    '''
    Main Image processing function to achieve grayscale puzzle without gridlines 
    '''

    if gray.ndim >= 3: gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    print("Begin ExtractPuzzle")

    # Begin Preprocess
    close = preprocess(gray, 19)

    '''Isolate the puzzle'''
    mask = np.zeros_like(close)

    _, contour,hier = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Largest contour 
    best_cnt = sorted(contour, key=cv2.contourArea, reverse=True)[0]
    print('Largest Contour FOUND')

    cv2.drawContours(mask,[best_cnt],0,255,-1)
    cv2.drawContours(mask,[best_cnt],0,0,2)

    # Isolate the puzzle to a black background
    res = cv2.bitwise_and(gray,mask)

    # Enlarge Puzzle to fit entire image
    warped = four_point_transform(res, best_cnt)
    print('Unwarping DONE!')

    # Find the sweet spot for thresholding
    new = repetitiveThreshold(warped, 15)
    print('Thresholding DONE!')

    # Dilate to enlarge the grids (Easier detection and removal)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(new, kernel)

    # Find all vertical and horizontal lines
    dst = cv2.Canny(dilated, 50, 150, None, 3)
    mask = np.zeros_like(dst)
    lines = cv2.HoughLinesP(dst, 1, np.pi/180, 100, minLineLength=100, maxLineGap=80)
    a,b,c = lines.shape
    for i in range(a):
        cv2.line(mask, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), 255, 3, cv2.LINE_AA)

    overlapped = cv2.bitwise_and(mask, dilated)

    # Perform floodfill on to cover where those lines are
    coords = np.argwhere(overlapped!=0)
    repetitiveFloodfill(dilated, overlapped, coords, 0)
    print('Floodfill DONE!')

    # Slice image in case borders are not properly removed
    dilated = dilated[5:-5, 5:-5]
    resized = cv2.resize(dilated, (1800,1800))

    print('Complete Puzzle Extraction')
    return resized


if __name__ == '__main__':

    photo = cv2.imread('puzzle.jpg', 0)

    photo = extractPuzzle(photo)
    puzzle2array(photo)
    photo = cv2.resize(photo, (250,400))
    view(photo, (400,400))
    # cv2.imwrite('./displayimages/thresholdpuzzle.jpg', photo)