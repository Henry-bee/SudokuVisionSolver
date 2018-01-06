from Solver import Sudoku


puzzle = Sudoku('puzzle', 'puzzle.jpg')

puzzle.solve()

print (puzzle.solution)