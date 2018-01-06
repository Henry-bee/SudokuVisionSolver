Sudoku Vision Solver (Python) 
=================================

## Description

Solving a Sudoku Puzzle visually! 

## Main Features
- Image Processing using OpenCV2
- Digit Recognition using Convolutional Neural Network in Tensorflow
- Backtracking algorithm

## Prerequisites

- Python3.5
- Tensorflow
- OpenCV2
- Numpy

## How to run
> Save a puzzle image as puzzle.jpg
> execute run.py

## Algorithm

1. Image Preprocessing (Gaussian Blur -> Adaptive Thresholding -> Dilate with cross kernels (darkens grid)) 
2. Find the largest Contour in the image ( Assuming puzzle is the largest blob and contour has four points) 
3. Unwarp the puzzle and fit to image
4. Selective Adaptive Threshold to create contrast between digits and background (Slowly increasing blocksize until only 15% white percentage is achieved.. Not the best method) 
5. Probabilistic Hough Line Transform to find the location of all grid lines 
6. Floodfill Algorithm to eliminate the grids
7. Result Image = White Digits + Black Background
8. Divide up the image into 81 sections
9. If section >5% white pixels, find largest connected component, isolate it and centralize
10. Feed section thru CovNet to predict digit
11. Solve puzzle using Backtrack Algorithm!

--------------------------------------

So here's a breakdown of what happened

First, input image <br />
![a](https://github.com/Henry-bee/SudokuVisionSolver/blob/master/displayimages/displaypuzzle.jpg) <br />

After Step 7<br />
![a](https://github.com/Henry-bee/SudokuVisionSolver/blob/master/displayimages/thresholdpuzzle.jpg) <br />

Meanwhile in the command line, <br />
![a](https://github.com/Henry-bee/SudokuVisionSolver/blob/master/displayimages/output_img1.png) <br />

![a](https://github.com/Henry-bee/SudokuVisionSolver/blob/master/displayimages/output_img2.png) <br />

.... it's solved!

## Sidenotes
------------------------

- Network is trained with only with images from "Big Book of Sudoku" 
- Puzzle with handwritten digits won't work
- Architecture of CovNet: COV30 -> MAXPOOL -> COV15 -> MAXPOOL -> DROPOUT -> FC128 -> FC50 -> FC10 
- 10 classes: digits (1-9) + noise
- Relatively poor at filtering out noisy cells

![a](https://github.com/Henry-bee/SudokuVisionSolver/blob/master/displayimages/noise1.jpg) <br />
Noise accumulated from the shadows of texts on the next page

![a](https://github.com/Henry-bee/SudokuVisionSolver/blob/master/displayimages/noise2.jpg) <br />
Result of poorly trimmed grid lines

- If you have a better way to filter out the noise or extract digits in a cleaner fashion, pls share! Any constructive feedback is welcome!:) 




