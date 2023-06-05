# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 84:
# 102843 Joao Melo
# 103597 Andre Santos

from sys import stdin
from search import (
    Problem,
    Node,
    depth_first_tree_search,
)

#consts
BOARD_SIZE=10

WATER = "."
DEFAULT_WATER = "W"
CIRCLE = "c"
TOP = "t"
MIDDLE = "m"
BOTTOM = "b"
LEFT = "l"
RIGHT = "r"
EMPTY = ""
PLACEHOLDER = "o"

#aux
is_water = lambda x: x and x == DEFAULT_WATER or x == WATER
is_circle = lambda x: x and x.lower() == CIRCLE
is_top = lambda x: x and x.lower() == TOP
is_middle = lambda x: x and x.lower() == MIDDLE
is_bottom = lambda x: x and x.lower() == BOTTOM
is_left = lambda x: x.lower() == LEFT
is_right = lambda x: x.lower() == RIGHT
is_empty = lambda x: x == EMPTY
is_border = lambda x: x == None
is_placeholder = lambda x: x == PLACEHOLDER

is_piece = lambda x: not (is_water(x) or is_border(x) or is_empty(x))

class BimaruState:
  state_id = 0

  def __init__(self, board):
    self.board = board
    self.id = BimaruState.state_id
    BimaruState.state_id += 1

  def __lt__(self, other):
    return self.id < other.id

class Board:  
  def __init__(self, rows_target, cols_target):
    self.rows_target = rows_target
    self.cols_target = cols_target

    self.state = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    self.boat_pieces = []
    self.left_boats =[4, 3, 3, 2, 2, 2, 1, 1, 1, 1]
    self.current_boats = []
    self.skip_coords = []

  def clean_board(self):
    #clean non empty rows/cols
    self.apply_targets()

    for piece in self.boat_pieces:
      row, col, obj = piece
      deductions = self.get_deductions(int(row), int(col), obj)

      #apply deductions
      for drow, dcol, dobj in deductions:
        self.place(drow, dcol, dobj)

  def place(self, row, col, obj):
    row, col = int(row), int(col)
    current_obj = self.get_value(row, col)

    # if values are not in the board boundary
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
      return

    # if we are overwriting something
    elif not is_empty(current_obj) and not obj == DEFAULT_WATER and not is_placeholder(current_obj):
      return

    #place the object
    self.state[row][col] = obj

    if is_water(obj): #if water skip the rest
      return

    #save the piece
    if is_placeholder(current_obj):
      #remove the old piece from the pieces list
      self.boat_pieces.remove([row, col, current_obj])
      
    #add the placed piece to the pieces list
    self.boat_pieces.append([row, col, obj])

    #subtract targets
    if is_empty(current_obj):
      self.subtract_targets(row, col)
      self.clean_surroundings(row, col, obj)

  def clean_surroundings(self, row, col, obj):
    #clean object surroundings
    surroundings = []
    if is_circle(obj):
      surroundings = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    elif is_top(obj):
      surroundings = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 1)]

    elif is_bottom(obj):
      surroundings = [(-1, -1), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    #TODO: if is middle is one block away from a border that block is not water (object)
    elif is_middle(obj) or is_placeholder(obj):
      surroundings = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    elif is_left(obj):
      surroundings = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (1, -1), (1, 0), (1, 1)]

    elif is_right(obj):
      surroundings = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for adjust_row, adjust_col in surroundings:
      new_row, new_col = row + adjust_row, col + adjust_col

      if is_empty(self.get_value(new_row, new_col)):
        self.place(new_row, new_col, WATER)

  def get_value(self, row, col):
    """Devolve o valor na respetiva posição do tabuleiro."""
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
      return self.state[row][col]

    return None
  
  def adjacent_vertical_values(self, row, col):
    """Devolve os valores imediatamente acima e abaixo,
    respectivamente."""
    top = self.get_value(row-1, col)
    bottom = self.get_value(row+1, col)

    return (top, bottom)

  def adjacent_horizontal_values(self, row, col):
    """Devolve os valores imediatamente à esquerda e à direita,
    respectivamente."""
    left = self.get_value(row, col-1)
    right = self.get_value(row, col+1)

    return (left, right)

  def distance_to_boundaries(self, row, col):
    '''
    returns the distance to the boundaries\n
    (top, left, right, bottom)
    '''
    return (row, col, BOARD_SIZE-1-col , BOARD_SIZE-1-row)

  def clean_row(self, row):
    for col in range(BOARD_SIZE):
      if is_empty(self.get_value(row, col)):
        self.place(row, col, WATER)

  def clean_col(self, col):
    for row in range(BOARD_SIZE):
      if is_empty(self.get_value(row, col)):
        self.place(row, col, WATER)

  def subtract_targets(self, row, col):
    self.rows_target[row]-= 1
    self.cols_target[col]-= 1

  def apply_targets(self):
    for row_index, target in enumerate(self.rows_target):
      if target == 0:
        self.clean_row(row_index)
      else:
        empty_spaces = self.empty_row_spaces(row_index)
        n_spaces = len(empty_spaces)
        if n_spaces and n_spaces == target:
          for col in empty_spaces:
            self.place(row_index, col, PLACEHOLDER)

    for col_index, target in enumerate(self.cols_target):
      if target == 0:
        self.clean_col(col_index)
      else:
        empty_spaces = self.empty_col_spaces(col_index)
        n_spaces = len(empty_spaces)
        if n_spaces and n_spaces == target:
          for row in empty_spaces:
            self.place(row, col_index, PLACEHOLDER)
        
  def empty_row_spaces(self, row):
    spaces = []
    for col in range(BOARD_SIZE):
      if is_empty(self.get_value(row, col)):
        spaces.append(col)

    return spaces
  
  def empty_col_spaces(self, col):
    spaces = []
    for row in range(BOARD_SIZE):
      if is_empty(self.get_value(row, col)):
        spaces.append(row)

    return spaces

  def get_deductions(self, row, col, obj):
    #init deductions array
    deductions = []

    row, col = int(row), int(col) 
    top_d, left_d, right_d, bottom_d = self.distance_to_boundaries(row, col)
    left_obj, right_obj = self.adjacent_horizontal_values(row, col)
    top_obj, bottom_obj = self.adjacent_vertical_values(row, col)

    if is_right(obj): 
      deductions.append((row, col-1, PLACEHOLDER))

    elif is_left(obj):
      deductions.append((row, col+1, PLACEHOLDER))

    elif is_bottom(obj):
      deductions.append((row-1, col, PLACEHOLDER))

    elif is_top(obj):
      deductions.append((row+1, col, PLACEHOLDER))
    
    elif is_middle(obj):
      #middle block is on top/bottom rows and is 1 block to the wall
      if top_d == 0 or bottom_d == 0 or is_water(top_obj) or is_water(bottom_obj):
        if not(left_d == 1 or is_water(self.get_value(row, col-2))):
          deductions.append((row, col-1, PLACEHOLDER))

        if not(right_d == 1 or is_water(self.get_value(row, col+2))):
          deductions.append((row, col+1, PLACEHOLDER))

      #middle block is on left/right columns and is 1 block away from top/bottom
      elif left_d == 0 or right_d == 0 or is_water(left_obj) or is_water(right_obj):
        deductions.append((row-1, col, PLACEHOLDER))

        if not(bottom_d == 1 or is_water(self.get_value(row+2, col))):
          deductions.append((row+1, col, PLACEHOLDER))

    #filter deductions
    filtered_deductions = [d for d in deductions if is_empty(self.get_value(d[0], d[1]))]
    return filtered_deductions

  def validate_boat(self, row, col, boat_size, direction, empty_blocks, placeholder_blocks):
    valid = True

    if boat_size == 0:
      return False

    #if there are more empty blocks than the target
    if direction == "bottom":
      if empty_blocks > self.cols_target[col]:
        return False
      
      for i in range(row, row+boat_size):
        if is_empty(self.get_value(i, col)) and self.rows_target[i] == 0:
          return False
        
    
    if direction == "right":
      if empty_blocks > self.rows_target[row]:
        return False
      
      for i in range(col, col+boat_size):
        if is_empty(self.get_value(row, i)) and self.cols_target[i] == 0:
          return False

    #confirmation edge cases
    if boat_size == 1:
      valid = self.is_one_boat(row, col)

    if boat_size == 2:
      if direction == "bottom":
        valid = self.is_two_boat_bottom(row, col)

      elif direction == "right":
        valid = self.is_two_boat_right(row, col)

    if boat_size == 3:
      if direction == "bottom":
        valid = self.is_three_boat_bottom(row, col)

      if direction == "right":
        valid = self.is_three_boat_right(row, col)

    if boat_size == 4:
      if direction == "bottom":
        valid = self.is_four_boat_bottom(row, col)

      if direction == "right":
        valid = self.is_four_boat_right(row, col)
      
    return valid

  def place_boat(self, row, col, size, direction):
    #remove boat size from boats_left
    if size not in self.left_boats:
      return
    
    self.left_boats.remove(size)

    if size == 1:
      self.place(row, col, CIRCLE)
    
    elif size == 2:
      if direction == "right":
        self.place(row, col, LEFT)
        self.place(row, col+1, RIGHT)
      elif direction == "bottom":
        self.place(row, col, TOP)
        self.place(row+1, col, BOTTOM)
    
    elif size == 3:
      if direction == "right":
        self.place(row, col, LEFT)
        self.place(row, col+1, MIDDLE)
        self.place(row, col+2, RIGHT)
      elif direction == "bottom":
        self.place(row, col, TOP)
        self.place(row+1, col, MIDDLE)
        self.place(row+2, col, BOTTOM)

    elif size == 4:
      if direction == "right":
        self.place(row, col, LEFT)
        self.place(row, col+1, MIDDLE)
        self.place(row, col+2, MIDDLE)
        self.place(row, col+3, RIGHT)
      elif direction == "bottom":
        self.place(row, col, TOP)
        self.place(row+1, col, MIDDLE)
        self.place(row+2, col, MIDDLE)
        self.place(row+3, col, BOTTOM)

    self.current_boats.append((row, col, size, direction))

    #add coords to skip
    for i in range(size):
      if direction == "right":
        self.skip_coords.append((row, col+i))
      elif direction == "bottom":
        self.skip_coords.append((row+i, col))

  def search_boats(self, size=None):
    if len(self.left_boats) == 0 or (size and size not in self.left_boats):
      return []
    
    boats = []
    max_boat_size = size or self.left_boats[0]

    for row in range(BOARD_SIZE):
      for col in range(BOARD_SIZE):

        #skip if coord is already used
        if (row, col) in self.skip_coords:
          continue

        for direction in ["right", "bottom"]: 
          boat_size = 0
          empty_blocks = 0
          placeholder_blocks = 0

          while boat_size < max_boat_size:
            #calculate new boat piece position 
            if direction == "right":
              new_col = col + boat_size
              new_row = row
            elif direction == "bottom":
              new_col = col
              new_row = row + boat_size
            
            #break case
            sp = self.get_value(new_row, new_col)
            if is_water(sp) or is_border(sp):  
              break
            
            #counting
            elif is_empty(sp):
              empty_blocks += 1
            elif is_placeholder(sp):
              placeholder_blocks += 1
      
            boat_size += 1

          #if boat doesnt have the size we want
          if boat_size != max_boat_size:
            continue

          valid = self.validate_boat(row, col, boat_size, direction, empty_blocks, placeholder_blocks)
          if valid:
            boats.append([row, col, boat_size, direction, empty_blocks, placeholder_blocks])

          #if we are searching for 1 size boats, we can stop here
          if max_boat_size == 1:
            break
    
    return boats

  def is_one_boat(self, row, col):
    v = self.get_value(row, col)
    
    if not (is_circle(v) or is_empty(v) or is_placeholder(v)):
      return False

    offset = [(-1,-1), (-1,0), (-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for i, j in offset:
      if is_piece(self.get_value(row+i, col+j)):
        return False

    return True

  def is_two_boat_right(self, row, col, gt=False):
    v1 = self.get_value(row, col)
    v2 = self.get_value(row, col+1)

    if not (is_left(v1) or is_empty(v1) or is_placeholder(v1)) or not (is_right(v2) or is_empty(v2) or is_placeholder(v2)):
      return False

    offset = [(-1,-1), (-1,0), (-1,1), (-1,2),
              (0,-1),(0,2),
              (1,-1),(1,0),(1,1), (1,2)]
    for i, j in offset:
      if is_piece(self.get_value(row+i, col+j)):
        return False    
    
    return True

  def is_two_boat_bottom(self, row, col, gt=False):
    v1 = self.get_value(row, col)
    v2 = self.get_value(row+1, col)

    if not (is_top(v1) or is_empty(v1) or is_placeholder(v1)) or not (is_bottom(v2) or is_empty(v2) or is_placeholder(v2)):
      return False

    offset = [(-1,-1), (-1,0), (-1,1),
              (0,-1),(0,1),
              (1,-1),(1,1),
              (2,-1),(2,0),(2,1)]
    for i, j in offset:
      if is_piece(self.get_value(row+i, col+j)):
        return False    
    
    return True

  def is_three_boat_right(self, row, col, gt=False):
    v1 = self.get_value(row, col)
    v2 = self.get_value(row, col+1)
    v3 = self.get_value(row, col+2)

    if not (is_left(v1) or is_empty(v1) or is_placeholder(v1)) or not (is_middle(v2) or is_empty(v2) or is_placeholder(v2)) or not (is_right(v3) or is_empty(v3) or is_placeholder(v3)):
      return False

    offset = [(-1,-1), (-1,0), (-1,1), (-1,2), (-1,3),
              (0,-1),(0,3),
              (1,-1),(1,0),(1,1), (1,2), (1,3)]
    for i, j in offset:
      if is_piece(self.get_value(row+i, col+j)):
        return False    
    
    return True

  def is_three_boat_bottom(self, row, col, gt=False):
    v1 = self.get_value(row, col)
    v2 = self.get_value(row+1, col)
    v3 = self.get_value(row+2, col)

    if not (is_top(v1) or is_empty(v1) or is_placeholder(v1)) or not (is_middle(v2) or is_empty(v2) or is_placeholder(v2)) or not (is_bottom(v3) or is_empty(v3) or is_placeholder(v3)):
      return False

    offset = [(-1,-1), (-1,0), (-1,1),
              (0,-1),(0,1),
              (1,-1),(1,1),
              (2,-1),(2,1),
              (3,-1),(3,0),(3,1)]
    for i, j in offset:
      if is_piece(self.get_value(row+i, col+j)):
        return False    
    
    return True

  def is_four_boat_right(self, row, col, gt=False):
    v1 = self.get_value(row, col)
    v2 = self.get_value(row, col+1)
    v3 = self.get_value(row, col+2)
    v4 = self.get_value(row, col+3)
    
    if not (is_left(v1) or is_empty(v1) or is_placeholder(v1)) or not (is_middle(v2) or is_empty(v2) or is_placeholder(v2)) or not (is_middle(v3) or is_empty(v3) or is_placeholder(v3)) or not (is_right(v4) or is_empty(v4) or is_placeholder(v4)):
      return False

    offset = [(-1,-1), (-1,0), (-1,1), (-1,2), (-1,3), (-1,4),
              (0,-1),(0,4),
              (1,-1),(1,0),(1,1), (1,2), (1,3), (1,4)]
    for i, j in offset:
      if is_piece(self.get_value(row+i, col+j)):
        return False    
    
    return True

  def is_four_boat_bottom(self, row, col, gt=False):
    v1 = self.get_value(row, col)
    v2 = self.get_value(row+1, col)
    v3 = self.get_value(row+2, col)
    v4 = self.get_value(row+3, col)
    
    if not (is_top(v1) or is_empty(v1) or is_placeholder(v1)) or not (is_middle(v2) or is_empty(v2) or is_placeholder(v2)) or not (is_middle(v3) or is_empty(v3) or is_placeholder(v3)) or not (is_bottom(v4) or is_empty(v4) or is_placeholder(v4)):
      return False

    offset = [(-1,-1), (-1,0), (-1,1),
              (0,-1),(0,1),
              (1,-1),(1,1),
              (2,-1),(2,1),
              (3,-1),(3,1),
              (4,-1),(4,0),(4,1)]
    for i, j in offset:
      if is_piece(self.get_value(row+i, col+j)):
        return False    
    
    return True


  def beauty_print(self):
    max_len = max(len(str(val)) for row in self.state for val in row)

    row_str = ""
    for ct in self.cols_target:
      val_str = str(ct)
      padding = " " * (max_len - len(val_str))
      row_str += val_str + padding + " "

    print(f"\n{row_str}", end="\n")
    print("-" * 21)

    matrix_str = ""
    for i in range(BOARD_SIZE):
        row_str = ""
        for j in range(BOARD_SIZE):
            val_str = str(self.state[i][j])
            padding = " " * (max_len - len(val_str))
            row_str += val_str + padding + " "
        matrix_str += f"{row_str}| {self.rows_target[i]}\n"

    return matrix_str

  def print_board(self):
    """Imprime o tabuleiro no standard output (stdout)."""
    for row in range(BOARD_SIZE):
      for col in range(BOARD_SIZE):
        if self.get_value(row, col) is None:
          print('!', end='')
        else:
          print(self.get_value(row, col), end='')
      print()

  def board_copy(self):
    """Retorna uma cópia do tabuleiro atual."""
    rows_target = self.rows_target.copy()
    cols_target = self.cols_target.copy()

    board_copy = Board(rows_target, cols_target)

    board_copy.state = [[self.state[row][col] for col in range(BOARD_SIZE)] for row in range(BOARD_SIZE)]
    board_copy.boat_pieces = self.boat_pieces.copy()
    board_copy.left_boats = self.left_boats.copy()
    board_copy.current_boats = self.current_boats.copy()
    board_copy.skip_coords = self.skip_coords.copy()

    return board_copy
  
  @staticmethod
  def parse_instance():
    """Lê o test do standard input (stdin) que é passado como argumento
    e retorna uma instância da classe Board.
    """
    rows_target = list(map(int, stdin.readline().split()[1:]))
    cols_target = list(map(int, stdin.readline().split()[1:]))

    board_instance = Board(rows_target, cols_target)

    #build the initial board
    pieces_placed = 0
    for _ in range(int(stdin.readline())):
      row, col, obj = stdin.readline().split()[1:]
      board_instance.place(int(row), int(col), obj)
      pieces_placed += 1

    #check what boats are already complete
    for size in range(min(4, pieces_placed), 0, -1):
      f = board_instance.search_boats(size=size)
      
      for boat in f:
        row, col, boat_size, direction, empty_blocks, placeholder_blocks = boat
        if empty_blocks == 0 and placeholder_blocks == 0:
          board_instance.left_boats.remove(boat_size)
          board_instance.current_boats.append((row, col, boat_size, direction))

          #add coords to skip
          for i in range(size):
            if direction == "right":
              board_instance.skip_coords.append((row, col+i))
            elif direction == "bottom":
              board_instance.skip_coords.append((row+i, col))

    board_instance.clean_board()
    return board_instance


class Bimaru(Problem):
    def __init__(self, board: Board):
      """O construtor especifica o estado inicial."""
      self.state = BimaruState(board)
      super().__init__(self.state)

    def actions(self, state: BimaruState):
      if len(state.board.left_boats) == 0 or -1 in state.board.rows_target or -1 in state.board.cols_target: #invalid state
        return []

      max_boat_size = state.board.left_boats[0]
      found_boats = state.board.search_boats(size=max_boat_size)

      actions = []
      for boat in found_boats:
        _, _, boat_size, _, empty_blocks, placeholder_blocks = boat

        #if the boat is already placed, ignore it
        if empty_blocks == 0 and placeholder_blocks == 0:
          continue

        actions.append(boat)

      #sort actions based on the amount of placeholders
      actions.sort(key=lambda x: ((x[2]-x[4]), x[5]), reverse=True)
      return actions
      


    def result(self, state: BimaruState, action):
      
      new_board = state.board.board_copy()

      row, col, boat_size, direction, _, _ = action
      new_board.place_boat(row, col, boat_size, direction)
      new_board.clean_board()

      new_state = BimaruState(new_board)
      return new_state


    def goal_test(self, state: BimaruState):
      return len(state.board.left_boats) == 0

    def h(self, node: Node):
      return -len(node.state.board.left_boats)



if __name__ == "__main__":
  board: Board = Board.parse_instance()

  problem: Problem = Bimaru(board)

  goal_node: Node = depth_first_tree_search(problem)

  goal_node.state.board.print_board()