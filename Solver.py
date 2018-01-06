import cv2
from vision import extractPuzzle, puzzle2array
from backtrack import solve, isSolvable
from vision_utils import view
import tensorflow as tf
import numpy as np


def top2(_list):
    
    _new_list = [(i, elem) for i, elem in enumerate(_list)]

    return sorted(_new_list, key= lambda x: x[1] ,reverse=True)[:2]
    
class Sudoku(object):

    def __init__(self, name, path):
        self.name = name
        self.originalImg = cv2.imread(path, 0)
        self.processedImg = extractPuzzle(self.originalImg)
        self.ImgArr = puzzle2array(self.processedImg)
        self.initializeModel()

    def initializeModel(self, path='./saved_model/sudoku_cv.ckpt.meta'):
        self.saver = tf.train.import_meta_graph(path)

        graph = tf.get_default_graph()
        self.input_x = graph.get_tensor_by_name("X:0")
        self.train = graph.get_tensor_by_name("dropout_flag:0")
        self.proba = graph.get_tensor_by_name('probability:0')
        self.prediction = graph.get_tensor_by_name("prediction_op:0")
    
    def img2digits(self):

        puzzle = []

        with tf.Session() as sess:
            self.saver.restore(sess,tf.train.latest_checkpoint('./saved_model/'))
            
            for r, row in enumerate(self.ImgArr):
                _row = []

                for c, cell in enumerate(row):
                
                    if len(cell) > 0:

                        x = cell
                        x = x[np.newaxis, :]
                        
                        x = x / 255.

                        digit, proba = sess.run( [self.prediction, self.proba], feed_dict={
                            self.input_x : x,
                            self.train: False
                        })

                        print ("Cell: %s%s, Digit: %s, probability:%s "%(r,c,digit[0], top2(proba[0]) ))

                        digit = digit[0]
                        proba = proba[0][digit]

                        digit = 0 if proba < 0.5 else digit
                        
                        _row.append(digit)

                    else:
                        _row.append(0)

                puzzle.append(_row)

        self.puzzle = np.array(puzzle)


    def solve(self):
        
        self.img2digits()

        if not isSolvable(self.puzzle):
            self.solution = self.puzzle
            print ("Not Solvable")

            self.writeCellsOut()
            return False

        solved, complete_puzzle = solve(self.puzzle)

        if solved:
            print ("Solved!")
            self.solution = complete_puzzle
            
        else:
            print ("Cant find solution")
            self.writeCellsOut()
            self.solution = self.puzzle
        
        return solved

    def writeCellsOut(self, PATH='./images/cells/%s_%s_%s.jpg'):

        for h, row in enumerate(self.ImgArr):
            for w, cell in enumerate(row):
                if len(cell) < 1: continue
                else:
                    cv2.imwrite(PATH%(self.name, h, w), cell)
