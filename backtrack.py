import numpy as np

def containDuplicate(array):
    '''
    Check if a list contains duplicate
    '''
    hashmap = [ 0 for i in range(10)]

    for elem in array:

        if elem == 0: continue

        hashmap[elem] += 1

        if hashmap[elem] > 1:
            return True

    return False


def isSolvable(puzzle):
    
    # Check for duplicates in a row
    rows, cols = puzzle.shape

    for r in range(rows):
        
        _row = puzzle[r, :].tolist()
        illegal = containDuplicate(_row)

        if illegal: return False
    
    # Check if theres duplicate in a column 
    for c in range(cols):

        _col = puzzle[:, c].tolist()
        illegal = containDuplicate(_col)

        if illegal: return False

    # Check Boxes for duplicates
    for i in [0, 3, 6]:
        for j in [0, 3, 6]:
            
            _box = puzzle[i:i+3, j:j+3].flatten().tolist()
            illegal = containDuplicate(_box)
        
        if illegal: return False

    return True


def findBox(r,c, puzzle):
    '''
    Returns the box which cell belongs to
    '''
    start = int(r / 3) *3
    end = int(c / 3) *3

    return puzzle[start:start+3,end:end+3].flatten()

def isSolved(puzzle):
    return (puzzle != 0).all()

def isLegal(r,c, candidate, puzzle):
    '''
    Check if this placement is legal, no conflict/duplicate
    Return True if legal, False otherwise
    '''
    used = []
    used += findBox(r, c, puzzle).tolist() # in the box
    used += puzzle[r, :].tolist() # Across this row
    used += puzzle[:, c].tolist() # Along the column
    
    return candidate not in set(used)

def findNextEmptyCell(puzzle):

    for r in range(0,9):
        for c in range(0,9):
            if puzzle[r,c] == 0:
                return r,c

def solve(puzzle):

    # If puzzle is solved
    if isSolved(puzzle):
        return True, puzzle 
    
    # Find the coordinate of the next empty cell
    row, col = findNextEmptyCell(puzzle)
     
    for candidate in range(1, 10):
        # is this candidate legal?
        if isLegal(row, col, candidate, puzzle):
            # Set this value
            puzzle[row,col] = candidate

            if solve(puzzle)[0]: 
                return True, puzzle
            else:
                puzzle[row, col] = 0 # Backtracking Step

    return False, puzzle

