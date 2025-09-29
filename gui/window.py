import tkinter as tk

from game.board import Board
from game.models import BOARD_SIZE, Worker, Cell
from gui.notation import coords_to_notation, GameNotation

CELL = 80 #pixels per cell
MARGIN = 20 #padding

def place(board, worker_id: str, owner: str, pos: tuple[int, int]) -> None: # create and place worker on board
  
    w = Worker(id=worker_id, owner=owner, pos=pos)
    board.workers.append(w)
    board.grid[pos].worker_id = worker_id

class SantoriniTk(tk.Tk):
    def __init__(self, board: Board):
        super().__init__()
        self.title("Santorini")
        self.board = board

        w = h = MARGIN * 2 + CELL * BOARD_SIZE
        self.canvas = tk.Canvas(self, width=w, height=h, bg="white")
        self.canvas.pack()

        self.status = tk.Label(self , text ="Board view", anchor = "w")
        self.status.pack(fill = "x")

        self.draw()

# Draw help

    def _rc_to_xy(self, r:int, c:int):
        x1 = MARGIN + c * CELL
        y1 = MARGIN + r * CELL
        x2 = x1+ CELL
        y2 = y1+ CELL
        return (x1, y1, x2, y2)
    
    def _draw_grid(self):
        for i in range(BOARD_SIZE+1):
            x = MARGIN + i * CELL
            y = MARGIN + i * CELL

            #vertical
            self.canvas.create_line(x, MARGIN, x, MARGIN + CELL * BOARD_SIZE)

            #horizontal
            self.canvas.create_line( MARGIN, y, MARGIN + CELL * BOARD_SIZE, y)

        # headers
        for c in range(BOARD_SIZE):
            x = MARGIN + c * CELL + CELL // 2
            self.canvas.create_text(x, MARGIN // 2, text = chr(ord('A') + c))

        for r in range(BOARD_SIZE):
            y = MARGIN + r * CELL + CELL // 2
            self.canvas.create_text(MARGIN // 2, y, text = str(r+1))

    def _draw_cells(self): #show height and worker id in cell
        for r in range (BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cell = self.board.grid[(r,c)]
                x1, y1, x2, y2 = self._rc_to_xy(r,c)

                # BOLD for height 
                self.canvas.create_text((x1+x2)//2, y1 + 14, text = str(cell.height), font = ("Arial", 12, "bold"))

                if cell.worker_id is not None:
                    self.canvas.create_text((x1+x2)//2, (y1+y2)//2, text = cell.worker_id, font = ("Arial", 11))

    def draw(self):
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_cells()

def main():

    
    board = Board([])

    place(board, "P1A", "P1", (0, 0))
    place(board, "P2A", "P2", (4, 4))
    board.grid[(0,0)].height = 2
    board.grid[(1,1)].height = 1
    board.grid[(2,2)].height = 3

    app = SantoriniTk(board)
    app.mainloop()

if __name__ == "__main__":
        main()