import tkinter as tk

from game.board import Board
from game.models import BOARD_SIZE, Worker, Cell
from gui.notation import coords_to_notation, GameNotation

from game.rules import legal_moves, legal_builds
from game.moves import move_worker, build_block


CELL = 80 #pixels per cell
MARGIN = 20 #padding

COLOR_MOVE = "lightblue"
COLOR_BUILD = "Orange"
COLOR_SELECTED = "lightgreen"

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

        
        #ui for interaction

        self.phase = "select_Worker"
        self.selected_worker = None # hold selected worker
        self.src = None # hold source cell for move/build
        self.legal = [] # hold legal target cells 

        self.canvas.bind("<Button-1>", self.on_click) #left click
        self.bind("<Escape>", lambda e: self.on_quit()) #esc to quit

        self.draw()


    def click_to_rc(self, event): #convert click to row/col

        x= event.x -MARGIN
        y= event.y -MARGIN

        if x<0 or y< 0 :
            return None
        
        c = x // CELL
        r = y // CELL
        if 0 <=r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            return int(r), int(c)
        return None #none if outside board
    

    def highlight(self,cells,outline = "blue"): #highlight legal cells
        for (r,c) in cells:
            x1, y1, x2, y2 = self._rc_to_xy(r,c)
            self.canvas.create_rectangle(x1,y1,x2,y2, outline = outline, width = 3)
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

    def draw(self, banner:str | None = None):
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_cells()

        if self.selected_worker and self.src:
            self.highlight_selected()

        if self.legal:
            color = COLOR_MOVE if self.phase == "select_dst" else COLOR_BUILD
            self.highlight(self.legal, outline = color)
    
    # Update status turn

        phase_text =self.phase.replace("_"," ")
        who = getattr(self.board, "current_player", "P1")
        self.status.config(text = banner or f"{who}: {phase_text}")

    def on_click(self, event): #handle click 
        rc = self.click_to_rc(event)
        if rc is None:
            return
        
        player =getattr(self.board, "current_player", "P1")

        #1select movable worker
        if self.phase == "select_Worker":

            picked = None
            for w in self.board.workers:
                if w.owner == player and w.pos == rc:
                    moves = legal_moves(self.board, w.pos)
                    if moves:
                        picked = w
                        self.legal= moves
                        break
            
            if picked:
                self.selected_worker = picked
                self.src = rc
                self.phase = "select_dst"
                self.draw(f"{player}: select move for {picked.id}")
            else:
                self.draw(f"{player}: no moves for worker ")
            return
        
        # click legal dst

        if self.phase == "select_dst":
            if rc in self.legal and self.selected_worker is not None:
                src = self.src
                dst = rc
                won = move_worker(self.board, self.selected_worker, dst)
                self.legal =[]
                self.phase = "select_build"
                self.draw(f"{player}: moved {src} -> {dst}")

                if won:
                    self.phase = "game_over"
                    self.draw(f"{player} wins by moving {self.selected_worker.id} to {coords_to_notation(dst)}!")
                    return

                self.legal = legal_builds(self.board, dst)
                self.phase = "select_build"
                self.legal = legal_builds(self.board, dst)
                self.draw(f"{player}: select a build square")

            else:
                self.draw(f"{player}: choose the highlighted cell")
            return

        # click legal build then end turn
        if self.phase == "select_build":
            if rc in self.legal and self.selected_worker is not None:
                build_block(self.board, self.selected_worker, rc)

                # end turn clear selection
                # 
                self.legal =[]
                self.selected_worker = None
                self.src = None
                self.phase = "select_Worker"

                # switch player
                other = "P2" if player == "P1" else "P1"
                self.board.current_player = other
                self.draw(f"Built at {coords_to_notation(rc)} -{other}'s turn")
            else:
                self.draw(f"{player}: choose the highlighted cell")
            return              

  
    
    def any_moves_for(self,player: str) -> bool:        #legal moves for player
        for w in self.board.workers:
            if w.owner == player and legal_moves(self.board, w.pos):
                return True
        return False
    def highlight_selected(self):
    
        if self.src is None:
            return
        x1, y1, x2, y2 = self._rc_to_xy(*self.src)
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=COLOR_SELECTED, width=4)

    def on_escape(self): #clear selection on escape
        if self.phase in {"select_dst", "select_build"}:
            self.phase = "select_worker"
            self.selected_worker = None
            self.src = None
            self.legal = []
            self.draw("Selection cleared")


def main():

    
    board = Board([])
    board.current_player = "P1"

    place(board, "P1A", "P1", (0, 0))
    place(board, "P2A", "P2", (4, 4))
    board.grid[(0,0)].height = 2
    board.grid[(1,1)].height = 1
    board.grid[(2,2)].height = 3

    app = SantoriniTk(board)
    app.mainloop()

if __name__ == "__main__":
        main()