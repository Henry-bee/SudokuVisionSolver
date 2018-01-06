Sudoku Vision Solver
=================================

## Description
-----------------------

Solving a Sudoku Puzzle visually! 

## Main Features
----------------------------
- Image Processing using OpenCV2
- Digit Recognition using Convolutional Neural Network in Tensorflow
- Backtracking algorithm

## Prerequisites
------------------------

- Python3.5
- Tensorflow
- OpenCV2
- Numpy

## How to run
-------------------------
> Save a puzzle image as puzzle.jpg
> execute run.py

## Algorithm
-------------------------

1. Image Preprocessing (Gaussian Blur -> Adaptive Thresholding -> Dilate with cross kernels (darkens grid)) 
2. Find the largest Contour in the image ( Assuming puzzle is the largest blob and contour has four points) 
3. Unwarp the puzzle and fit to image
4. Selective Adaptive Threshold to create contrast between digits and background (Slowly increasing blocksize until only 15% white percentage is achieved.. Not the best method) 
5. Probabilistic Hough Line Transform to find the location of all grid lines 
6. Floodfill Algorithm to eliminate the grids
7. Result Image = White Digits + Black Background
8. Divide up the image into 81 sections
9. In each section with >5% white pixels, find largest connected component, isolate it and centralize
10. Feed section thru CovNet to predict digit
11. Solve puzzle using Backtrack Algorithm!

The output should look like this


## Sidenotes
------------------------

- Network is trained with only with images from "Big Book of Sudoku" 
- Puzzle with handwritten digits won't perform that well 
- 



Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

## Tests

Describe and show how to run the tests with code examples.

## Contributors

Let people know how they can dive into the project, include important links to things like issue trackers, irc, twitter accounts if applicable.

## License

A short snippet describing the license (MIT, Apache, etc.)