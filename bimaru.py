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
  a = {4: []}
  def empty_pieces_row_col():
    pass

  def possible_placements():
    pass


  def clean_board(self):
    #predicts
    x = 0
    while True:
      stop = True
      print(f"ITER {x}\n")
      #clean non empty rows/cols
      print(f"CLEAN NON EMPTY ROWS/COLS")
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
      
      x+= 1
      print(self)

      if stop: #break out of the while True
        break
      
  def place(self, row, col, obj):
    row, col = int(row), int(col)

    # if values are not in the board boundary
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
      return

    # if we are overwriting something
    elif not is_empty(self.get_value(row, col)) and not obj == DEFAULT_WATER:
      return

    #place the object
    self.state[row, col] = obj

    if is_water(obj): #if water skip the rest
      return

    #save the piece
    self.boat_pieces = np.append(self.boat_pieces, [[row, col, obj]], axis=0)

    #subtract targets
    self.subtract_targets(row, col)

    #clean surroundings
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


  #TODO: ADVANCED TARGET CHECK AND CLEAN
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


  def is_boat(self, row: str, col: str):

    row, col = int(row), int(col)
    cur_val = self.get_value(row, col)

    if is_circle(cur_val):
      return True

    elif is_top(cur_val):
      m1 = is_middle(self.get_value(row+1, col))
      m2 = is_middle(self.get_value(row+2, col))
      b1 = is_bottom(self.get_value(row+1, col))
      b2 = is_bottom(self.get_value(row+2, col))
      b3 = is_bottom(self.get_value(row+3, col))

      if b1:
        return True
      elif m1 and b2:
        return True
      elif m1 and m2 and b3:
        return True

    elif is_bottom(cur_val):
      m1 = is_middle(self.get_value(row-1, col))
      m2 = is_middle(self.get_value(row-2, col))
      t1 = is_top(self.get_value(row-1, col))
      t2 = is_top(self.get_value(row-2, col))
      t3 = is_top(self.get_value(row-3, col))

      if t1:
        return True
      elif m1 and t2:
        return True
      elif m1 and m2 and t3:
        return True
    #L X X M
    elif is_left(cur_val):
      m1 = is_middle(self.get_value(row, col+1))
      m2 = is_middle(self.get_value(row, col+2))
      r1 = is_right(self.get_value(row, col+1))
      r2 = is_right(self.get_value(row, col+2))
      r3 = is_right(self.get_value(row, col+3))

      if r1:
        return True
      elif m1 and r2:
        return True
      elif m1 and m2 and r3:
        return True

    elif is_right(cur_val):
      m1 = is_middle(self.get_value(row, col-1))
      m2 = is_middle(self.get_value(row, col-2))
      l1 = is_left(self.get_value(row, col-1))
      l2 = is_left(self.get_value(row, col-2))
      l3 = is_left(self.get_value(row, col-3))

      if l1:
        return True
      elif m1 and l2:
        return True
      elif m1 and m2 and l3:
        return True

    return False


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
    board_instance.updated = False

    print(board_instance.rows_target)

    for _ in range(int(stdin.readline())):
      row, col, obj = stdin.readline().split()[1:]
      board_instance.place(int(row), int(col), obj)

      #TODO: calculate what piece is a boat and add them to open_pieces

    return board_instance


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        pass

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        pass

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO
        pass

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    board = Board.parse_instance()
    print(board)
    board.clean_board()

    with open("t.out", "w") as f:
      board = board.state
      for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
          f.write(f"{board[row,col]} ")
        f.write("\n")