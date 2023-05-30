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

#constantes
is_water = lambda x: x == "W" or x == "."
is_circle = lambda x: x.lower() == "c"
is_top = lambda x: x.lower() == "t"
is_middle = lambda x: x.lower() == "m"
is_bottom = lambda x: x.lower() == "b"
is_left = lambda x: x.lower() == "l"
is_right = lambda x: x.lower() == "r"
is_empty = lambda x: x == "|"

WATER = "."
CIRCLE = "c"
TOP = "t"
MIDDLE = "m"
BOTTOM = "b"
LEFT = "l"
RIGHT = "r"
EMPTY = "|"


class BimaruState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    def __init__(self, hints, rows_target, cols_target) -> None: 
        self.state = np.full((10,10), EMPTY)
        self.hints = hints
        self.rows_target = rows_target
        self.cols_target = cols_target

        #build the board
        self.build()

    def build(self):
        #place the hints in the board
        for hint in self.hints:
            row, col, object = hint

            #place
            self.place(row, col, object)
        
    """Representação interna de um tabuleiro de Bimaru."""
    def get_value(self, row: int, col: int) -> str | None:
        """Devolve o valor na respetiva posição do tabuleiro."""
        if 0 <= row < 10 and 0 <= col < 10:
            return self.state[row, col]

        return None

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        v1 = self.get_value(row, col-1)
        v2 = self.get_value(row, col+1)

        return (v1, v2) 
    
    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        h1 = self.get_value(row-1, col)
        h2 = self.get_value(row+1, col)

        return (h1, h2)

    def decrease_count(self, row, col):
        self.rows_target[row]-= 1
        self.cols_target[col]-= 1

        if self.rows_target[row] == 0:
          self.clean_row(row)

        if self.cols_target[col] == 0:
          self.clean_col(col)

    def place(self, row, col, obj):
        row, col = int(row), int(col)
        cur_value = self.get_value(row, col)

        #TODO: check if we are not placing adjacent to another object
        if not is_empty(cur_value):
            raise ValueError(f"Place is {cur_value} at ({row},{col})")
 
        if is_water(obj):
            self.state[row,col] = obj
            return 
        
        #clean object surroundings
        adjustments = []
        if is_circle(obj):
            adjustments = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        elif is_top(obj):
            adjustments = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 1)]

        elif is_bottom(obj):
            adjustments = [(-1, -1), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        #TODO: if is middle is one block away from a border that block is not water (object)
        elif is_middle(obj):
            adjustments = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        elif is_left(obj):
            adjustments = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (1, -1), (1, 0), (1, 1)]

        elif is_right(obj):
            adjustments = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for adjust_row, adjust_col in adjustments:
            new_row, new_col = row + adjust_row, col + adjust_col

            # Check if values are less than 0
            if not (0 <= new_row < 10 and 0 <= new_col < 10):
                continue
            
            # check if we are not overriding a value
            if is_empty(self.get_value(new_row, new_col)):
              self.place(new_row, new_col, WATER)

        #place the object in the board 
        self.state[row,col] = obj

        #decrease the count of the target and clean 0 target row/cols
        self.decrease_count(row, col)

    def clean_row(self, row):
      for col in range(10):
        if is_empty(self.get_value(row, col)):
          self.place(row, col, WATER)
    
    def clean_col(self, col):
        for row in range(10):
            if is_empty(self.get_value(row, col)):
              self.place(row, col, WATER)
    
    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """
        rows_target = np.fromiter(map(int, stdin.readline().split()[1:]), dtype=int)
        cols_target = np.fromiter(map(int, stdin.readline().split()[1:]), dtype=int)
        hints = np.empty((0, 3), dtype=object)

        for _ in range(int(stdin.readline())):
            r, c, value = stdin.readline().split()[1:]
            hint_row = np.array([[r, c, value]], dtype=str)
            hints = np.append(hints, hint_row, axis=0)   

        return Board(hints, rows_target, cols_target)



class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
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
    init_board = Board.parse_instance()
    with open("t.out", "w") as f:
      board = init_board.state
      for row in range(10):
        for col in range(10):
          f.write(f"{board[row,col]} ")
        f.write("\n")