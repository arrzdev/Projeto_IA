# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

#import sys
from sys import stdin
import numpy as np

from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
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
is_water = lambda x: x == DEFAULT_WATER or x == WATER
is_circle = lambda x: x.lower() == CIRCLE
is_top = lambda x: x.lower() == TOP
is_middle = lambda x: x.lower() == MIDDLE
is_bottom = lambda x: x.lower() == BOTTOM
is_left = lambda x: x.lower() == LEFT
is_right = lambda x: x.lower() == RIGHT
is_empty = lambda x: x == EMPTY
is_boat_piece = lambda x: is_bottom(x) or is_circle(x) or is_left(x) or is_middle(x) or is_right(x) or is_top(x) or is_placeholder(x)
is_border = lambda x: x == None
is_placeholder = lambda x: x == PLACEHOLDER


class BimaruState:
  state_id = 0

  def __init__(self, board):
    self.board = board
    self.id = BimaruState.state_id
    BimaruState.state_id += 1

  def __lt__(self, other):
    return self.id < other.id

class Board:
  def clean_board(self):
    #predicts
    while True:
      stop = True

      #clean non empty rows/cols
      self.apply_targets()

      for piece in self.boat_pieces:
        row, col, obj = piece
        deductions = self.get_deductions(row, col, obj)
        
        #if some deduction was found we need to iterate again
        if len(deductions) or self.updated:
          self.updated = False
          stop = False

        for drow, dcol, dobj in deductions:
          #place the deduction
          self.place(drow, dcol, dobj)

      if stop: #break out of the while True
        break

    self.resolve_boats()
      
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
    self.state[row, col] = obj

    if is_water(obj): #if water skip the rest
      return

    #save the piece
    if is_placeholder(obj):
      #remove the old piece from the pieces list
      self.boat_pieces = np.delete(self.boat_pieces, np.where((self.boat_pieces == [row, col, current_obj]).all(axis=1)), axis=0)
      
    #add the placed piece to the pieces list
    self.boat_pieces = np.append(self.boat_pieces, [[row, col, obj]], axis=0)

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

  def get_value(self, row: int, col: int) -> str | None:
    """Devolve o valor na respetiva posição do tabuleiro."""
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
      return self.state[row, col]

    return None
  
  def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
    """Devolve os valores imediatamente acima e abaixo,
    respectivamente."""
    top = self.get_value(row-1, col)
    bottom = self.get_value(row+1, col)

    return (top, bottom)

  def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
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
        self.updated = True

  def clean_col(self, col):
    for row in range(BOARD_SIZE):
      if is_empty(self.get_value(row, col)):
        self.place(row, col, WATER)
        self.updated = True
  

  def subtract_targets(self, row, col):
    self.rows_target[row]-= 1
    self.cols_target[col]-= 1

  def apply_targets(self):
    #TODO: analize the following complexity
    
    
    for row_index, target in enumerate(self.rows_target):
      if target == 0:
        self.clean_row(row_index)

    #apply for zero cols
    for col_index, target in enumerate(self.cols_target):
      if target == 0:
        self.clean_col(col_index)

    #apply for non zero rows
    for row_index, target in enumerate(self.rows_target):
      empty_spaces = self.empty_row_spaces(row_index)
      n_spaces = len(empty_spaces)
      if n_spaces and n_spaces == target:
        for col in empty_spaces:
          self.place(row_index, col, PLACEHOLDER)

    #apply for non zero values  
    for col_index, target in enumerate(self.cols_target):
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

      # if self.rows_target[row+1] - 1 == 0: #-1 is the ""placeholder""
      #   self.clean_row(row+1, [col])
    
    elif is_middle(obj):
      #middle block is on top/bottom rows and is 1 block to the wall
      if top_d == 0 or bottom_d == 0:
        deductions.append((row, col-1, PLACEHOLDER))
        deductions.append((row, col+1, PLACEHOLDER))

      #middle block is on left/right columns and is 1 block away from top/bottom
      elif left_d == 0 or right_d == 0:
        deductions.append((row-1, col, PLACEHOLDER))
        deductions.append((row+1, col, PLACEHOLDER))

      #middle block has non-empty pieces in the left/right side
      elif not ((is_empty(left_obj) or is_placeholder(left_obj)) and (is_empty(right_obj) or is_placeholder(right_obj))):
        deductions.append((row-1, col, PLACEHOLDER))
        deductions.append((row+1, col, PLACEHOLDER))

      #middle block has non-empty pieces in the top/bottom side
      elif not ((is_empty(top_obj) or is_placeholder(top_obj)) and (is_empty(bottom_obj) or is_placeholder(bottom_obj))):
        deductions.append((row, col-1, PLACEHOLDER))
        deductions.append((row, col+1, PLACEHOLDER))
    #filter deductions
    filtered_deductions = [d for d in deductions if is_empty(self.get_value(d[0], d[1]))]
    return filtered_deductions


  def resolve_boats(self):
    """
    This function find boats in the board and resolve them
    Either just removing them from the left_boats or also replacing placeholders
    """
    skip = []

    for row in range(self.state.shape[0]):
      for col in range(self.state.shape[1]):
        #skip
        if (row, col) in skip:
          continue
        
        #get the current max size boat we are looking for
        if len(self.left_boats) == 0:
          return
        
        #get max size boat
        max_size_boat = max(self.left_boats)

        #search boat to the right and to the bottom
        boat_direction = "right"
        boat_size = 0
        confirmed = False
        has_placeholder = False

        #we will search a boat 
        for search_direction in ["right", "bottom"]:
          size_search, confirmation_search, search_has_placeholder = self.search_boat(row, col, search_direction, max_size_boat)
          if size_search > boat_size:
            boat_size = size_search
            confirmed = confirmation_search
            boat_direction = search_direction
            has_placeholder = search_has_placeholder

        #place the boat
        if confirmed:
          #if there is placeholders in the boat, resolve them
          if has_placeholder:
            self.place_boat(row, col, boat_size, boat_direction)

          #remove the boat size from the list of left boats
          self.left_boats = np.delete(self.left_boats, np.where(self.left_boats == boat_size)[0][0])
          
          #mark the board cords as visited
          for i in range(boat_size):
            if boat_direction == "right":
              skip.append((row, col+i))
            elif boat_direction == "bottom":
              skip.append((row+i, col))         

  def place_boat(self, row, col, size, direction):
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

  def search_boat(self, row, col, direction, max_boat_size=4):
    """
    returns boat: [row, col, size], confirmed_boat or false
    """

    confirmed = True
    has_placeholder = False
    boat_size = 0

    #vertical search
    if direction == "bottom":
      while boat_size < max_boat_size:
        sp = self.get_value(row+boat_size, col)
        if is_water(sp) or is_border(sp):  
          break
        elif is_empty(sp):
          confirmed = False
        elif is_placeholder(sp):
          has_placeholder = True
        boat_size += 1
        

    #horizontal search
    elif direction == "right":
      while boat_size < max_boat_size:
        sp = self.get_value(row, col+boat_size)
        if is_water(sp) or is_border(sp): 
          break
        elif is_empty(sp):
          confirmed = False
        elif is_placeholder(sp):
          has_placeholder = True
        boat_size += 1

    #update confirmation on edge case
    if boat_size == 1:
      #get surrounding of the search position
      vv = self.adjacent_vertical_values(row, col)
      hv = self.adjacent_horizontal_values(row, col)

      if not (is_water(vv[0]) or is_border(vv[0])) or not (is_water(vv[1]) or is_border(vv[1]))\
          or not (is_water(hv[0]) or is_border(hv[0])) or not (is_water(hv[1]) or is_border(hv[1])):
        confirmed = False

    return boat_size, confirmed, has_placeholder


  def __str__(self):
    rows, cols = self.state.shape
    max_len = np.max([len(str(val)) for val in self.state.flatten()])

    row_str = ""
    for ct in self.cols_target:
      val_str = str(ct)
      padding = " " * (max_len - len(val_str))
      row_str += val_str + padding + " "

    print(f"\n{row_str}", end="\n")
    print("-" * 21)

    matrix_str = ""
    for i in range(rows):
        row_str = ""
        for j in range(cols):
            val_str = str(self.state[i, j])
            padding = " " * (max_len - len(val_str))
            row_str += val_str + padding + " "
        matrix_str += f"{row_str}| {self.rows_target[i]}\n"

    return matrix_str

  @staticmethod
  def parse_instance():
    """Lê o test do standard input (stdin) que é passado como argumento
    e retorna uma instância da classe Board.
    """
    board_instance = Board()
    board_instance.rows_target = np.fromiter(map(int, stdin.readline().split()[1:]), dtype=int)
    board_instance.cols_target = np.fromiter(map(int, stdin.readline().split()[1:]), dtype=int)
    board_instance.state = np.full((BOARD_SIZE,BOARD_SIZE), EMPTY)
    board_instance.boat_pieces = np.empty((0, 3), dtype=object)
    board_instance.left_boats = np.fromiter([4, 3, 3, 2, 2, 2, 1, 1, 1, 1], dtype=int)
    board_instance.updated = False

    #build the initial board
    for _ in range(int(stdin.readline())):
      row, col, obj = stdin.readline().split()[1:]
      board_instance.place(int(row), int(col), obj)

    return board_instance


class Bimaru(Problem):
    def __init__(self, board: Board):
      """O construtor especifica o estado inicial."""
      super().__init__(BimaruState(board))

    def actions(self, state: BimaruState):
      """Retorna uma lista de ações que podem ser executadas a
      partir do estado passado como argumento."""
      return []

    def result(self, state: BimaruState, action):
      """Retorna o estado resultante de executar a 'action' sobre
      'state' passado como argumento. A ação a executar deve ser uma
      das presentes na lista obtida pela execução de
      self.actions(state)."""
      # TODO
      pass

    def goal_test(self, state: BimaruState):
      for row in state.board.state:
        for col in row:
          if col == EMPTY:
            return False
      
      #?????
      for row in state.board.rows_target:
        if row != 0:
          return False
      
      #?????
      for col in state.board.cols_target:
        if col != 0:
          return False

      return True

    def h(self, node: Node):
      """Função heuristica utilizada para a procura A*."""
      # TODO
      pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
  
  board = Board.parse_instance()
  board.clean_board()

  problem = Bimaru(board)

  goal_node: Node = depth_first_tree_search(problem)

  if goal_node:
    print("SOLVED")
    print(goal_node.state.board)
  
  # for row in board.state:
  #   for col in row:
  #     print(col, end="")
  #   print()