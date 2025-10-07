import tkinter as tk
from tkinter import ttk
from typing import Dict, Any

from game.board import Board
from game.config import GameConfig
from game.models import BOARD_SIZE, Worker
from gui.notation import coords_to_notation
from ai.agent import Agent
from gui.gameplay import GameController


CELL = 80 #pixels per cell
MARGIN = 20 #padding

COLOR_MOVE = "lightblue"
COLOR_BUILD = "Orange"
COLOR_SELECTED = "lightgreen"

PLAYER_COLORS = {
    "P1": "#FF6B6B",    # Red
    "P2": "#4ECDC4",    # Teal
    "P3": "#45B7D1",    # Blue
}

def place(board, worker_id: str, owner: str, pos: tuple[int, int]) -> bool:
    """Create and place worker on board"""
    try:
        w = Worker(id=worker_id, owner=owner, pos=pos)
        board.workers.append(w)
        board.grid[pos].worker_id = worker_id
        return True
    except Exception as e:
        print(f"Error placing worker {worker_id}: {e}")
        return False

def place_workers_for_setup(board: Board, game_config: GameConfig) -> None:
    """Place workers in starting positions based on number of players"""
    if game_config.num_players == 2:
        # 2-player starting positions
        positions = [
            ("P1A", "P1", (0, 0)), ("P1B", "P1", (0, 2)),
            ("P2A", "P2", (4, 4)), ("P2B", "P2", (4, 2))
        ]
    elif game_config.num_players == 3:
        # 3-player starting positions (spread around board)
        positions = [
            ("P1A", "P1", (0, 0)), ("P1B", "P1", (0, 1)),
            ("P2A", "P2", (4, 3)), ("P2B", "P2", (4, 4)),
            ("P3A", "P3", (2, 2)), ("P3B", "P3", (1, 3))
        ]
    else:
        return  # Unsupported number of players

    for worker_id, owner, pos in positions:
        # Find the worker object and set its position
        worker = next((w for w in board.workers if w.id == worker_id), None)
        if worker:
            worker.pos = pos
            board.grid[pos].worker_id = worker_id


def choose_mode_ui() -> Dict[str, Any]:
    """Enhanced mode selection with 2-player and 3-player options"""
    root = tk.Tk()
    root.title("Choose Game Mode")
    root.geometry("400x300")
    root.resizable(False, False)

    # Number of players selection
    ttk.Label(root, text="Number of Players:", padding=10, font=("Arial", 12, "bold")).pack(anchor="w")

    players_var = tk.StringVar(value="2")
    player_frame = ttk.Frame(root)
    player_frame.pack(anchor="w", padx=20)

    ttk.Radiobutton(player_frame, text="2 Players", variable=players_var, value="2", padding=5).pack(anchor="w")
    ttk.Radiobutton(player_frame, text="3 Players", variable=players_var, value="3", padding=5).pack(anchor="w")

    # Game mode selection
    ttk.Label(root, text="Game Mode:", padding=10, font=("Arial", 12, "bold")).pack(anchor="w")

    mode_var = tk.StringVar(value="pvai")
    mode_frame = ttk.Frame(root)
    mode_frame.pack(anchor="w", padx=20)

    ttk.Radiobutton(mode_frame, text="Human vs Human", variable=mode_var, value="pvp", padding=5).pack(anchor="w")
    ttk.Radiobutton(mode_frame, text="Human vs AI", variable=mode_var, value="pvai", padding=5).pack(anchor="w")
    ttk.Radiobutton(mode_frame, text="AI vs AI", variable=mode_var, value="aivai", padding=5).pack(anchor="w")

    selected = {"val": None}

    def start():
        num_players = int(players_var.get())
        mode = mode_var.get()

        selected["val"] = {
            "num_players": num_players,
            "mode": mode
        }
        root.destroy()

    ttk.Button(root, text="Start Game", command=start, padding=10).pack(pady=20)

    root.mainloop()
    return selected["val"]


def build_players(mode_selection: Dict[str, Any], game_config: GameConfig) -> Dict[str, Dict]:
    """Build player configuration based on selected mode and game config"""
    mode = mode_selection["mode"]
    num_players = game_config.num_players

    players = {}

    if mode == "pvp":
        # All human players
        for i in range(num_players):
            player_id = game_config.get_player_id(i)
            players[player_id] = {"type": "HUMAN", "agent": None}

    elif mode == "pvai":
        # First player human, rest AI
        for i in range(num_players):
            player_id = game_config.get_player_id(i)
            if i == 0:
                players[player_id] = {"type": "HUMAN", "agent": None}
            else:
                players[player_id] = {"type": "AI", "agent": Agent(player_id)}

    elif mode == "aivai":
        # All AI players
        for i in range(num_players):
            player_id = game_config.get_player_id(i)
            players[player_id] = {"type": "AI", "agent": Agent(player_id)}

    return players

class SantoriniTk(tk.Tk):
    def __init__(self, board: Board, controller: GameController, game_config: GameConfig):
        super().__init__()
        self.title("Santorini")
        self.board = board
        self.game_over = False
        self.controller = controller
        self.game_config = game_config

        w = h = MARGIN * 2 + CELL * BOARD_SIZE
        self.canvas = tk.Canvas(self, width=w, height=h, bg="white")
        self.canvas.pack()

        # Status and player info frame
        info_frame = tk.Frame(self)
        info_frame.pack(fill="x")

        self.status = tk.Label(info_frame, text="Board view", anchor="w")
        self.status.pack(fill="x")

        # Player turn indicator
        self.player_info_frame = tk.Frame(info_frame)
        self.player_info_frame.pack(fill="x", pady=5)

        self.create_player_indicators()

        #ui for interaction

        self.phase = "select_Worker"
        self.selected_worker = None
        self.src = None
        self.legal = []

        self.canvas.bind("<Button-1>", self.on_click)
        self.bind("<Escape>", lambda e: self.on_escape())

        self.draw()

        # Start AI if needed
        if self.controller.is_ai_turn():
            self.after(50, self.ai_pump)
        # If the game starts with an AI (AI vs AI or AI vs P2)off the loop:
        if self.controller.is_ai_turn() and self.phase == "setup":
            self.after(50, self.ai_setup)   # pass the function, don't call it


    def ai_setup(self):
        if not self.controller.is_ai_turn():
            return None

        pid = getattr(self.board, 'current_player', 'P1')
        agent = self.controller.players[pid]["agent"]

        have = sum(1 for w in self.board.workers if w.owner == pid)
        if have >= 2:
            self.setup_workers()
            return

        label = "A" if have == 0 else "B"
        candidates = agent.setup_workers(self.board)

        pos = next((p for p in candidates if self.cell_empty(*p)), None)
        if pos is None:

            empties = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if self.cell_empty(r, c)]
            if not empties:
                return
            pos = empties[0]

        wid = f"{pid}{label}"
        placed =place(self.board, wid, pid, pos)

        if placed:
            self.draw(f"{pid}: workers placed")

        else:
            self.setup_workers() #next setup or start game

        self.setup_workers() #next setup or start game


    def cell_empty(self, r:int, c:int) -> bool: #check if cell is empty
        return self.board.grid[(r,c)].worker_id is None

    def setup_workers(self):
        pid = self.board.current_player
        have = sum(1 for w in self.board.workers if w.owner == pid)

    # If this player hasn't placed 2 yet, stay on this player
        if have < 2:
            self.setup_label = "A" if have == 0 else "B"
            msg = "place worker A" if have == 0 else "place second worker (B)"
            self.draw(f"{pid}: {msg}")
            if self.controller.is_ai_turn():
                self.after(50, self.ai_setup)   # schedule AI to place ONE worker
            return

    # This player now has 2 -> switch to the other player
        self.controller.end_turn()
        pid = self.board.current_player

    # If both sides are fully placed, start the game
        p1_done = sum(1 for w in self.board.workers if w.owner == "P1") == 2
        p2_done = sum(1 for w in self.board.workers if w.owner == "P2") == 2

        if p1_done and p2_done:
            self.phase = "select_Worker"
            self.draw(f"{pid}: select worker")
            if self.controller.is_ai_turn():
                self.after(50, self.ai_pump)
                return

    def create_player_indicators(self):
        """Create visual indicators for all players"""
        self.player_labels = {}

        for i in range(self.game_config.num_players):
            player_id = self.game_config.get_player_id(i)

            frame = tk.Frame(self.player_info_frame, relief="raised", borderwidth=2)
            frame.pack(side="left", padx=5, pady=2)

            color = PLAYER_COLORS.get(player_id, "gray")

            # Player color indicator
            color_label = tk.Label(frame, text="●", fg=color, font=("Arial", 20))
            color_label.pack(side="left")

            # Player info
            player_type = self.controller.players[player_id]["type"]
            text = f"{player_id} ({player_type})"
            label = tk.Label(frame, text=text, font=("Arial", 10))
            label.pack(side="left", padx=(5, 10))

            self.player_labels[player_id] = frame

    def update_current_player_display(self):
        """Highlight current player in the indicator"""
        for player_id, frame in self.player_labels.items():
            if player_id == self.board.current_player:
                frame.config(bg="yellow", relief="raised", borderwidth=3)
            else:
                frame.config(bg="SystemButtonFace", relief="raised", borderwidth=1)

    def _draw_cells(self):
        """Enhanced cell drawing with player colors"""
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cell = self.board.grid[(r, c)]
                x1, y1, x2, y2 = self._rc_to_xy(r, c)

                # Draw height
                self.canvas.create_text((x1+x2)//2, y1 + 14, text=str(cell.height),
                                      font=("Arial", 12, "bold"))

                # Draw worker with player color
                if cell.worker_id is not None:
                    worker = next((w for w in self.board.workers if w.id == cell.worker_id), None)
                    if worker:
                        color = PLAYER_COLORS.get(worker.owner, "black")
                        self.canvas.create_text((x1+x2)//2, (y1+y2)//2,
                                              text=cell.worker_id,
                                              font=("Arial", 10, "bold"),
                                              fill=color)

    def draw(self, banner: str | None = None):
        """Enhanced draw with player turn indicator"""
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_cells()

        if self.selected_worker and self.src:
            self.highlight_selected()

        if self.legal:
            color = COLOR_MOVE if self.phase == "select_dst" else COLOR_BUILD
            self.highlight(self.legal, outline=color)

        # Update player display
        self.update_current_player_display()

        # Update status
        phase_text = self.phase.replace("_", " ")
        who = self.board.current_player
        player_type = self.controller.players[who]["type"]

        if banner:
            self.status.config(text=banner)
        else:
            self.status.config(text=f"{who} ({player_type}): {phase_text}")

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

    def on_click(self, event): #handle click

        if self.game_over or self.controller.is_ai_turn():# ignore clicks during AI turn or after game end
            return
        rc = self.click_to_rc(event)
        if rc is None:
            return

        player = self.board.current_player

        if self.phase == "setup":

            if sum(1 for w in self.board.workers if w.owner == player) >= 2:
                self.draw(f"{player}: you already placed 2 workers")
                return
    # human placing workers
            if not self.cell_empty(*rc):
                self.draw(f"{player}: cell occupied—choose another")
                return

    # place worker A or B for the current (human) player
            wid = f"{player}{self.setup_label}"
            ok = place(self.board, wid, player, rc)
            if not ok:
                self.draw(f"{player}: invalid placement")
                return

            self.draw(f"{player}: placed {wid} at {coords_to_notation(rc)}")


            self.setup_workers()
            return

        #1select movable worker
        if self.phase == "select_Worker":

            picked = None
            for w in self.board.workers:
                if w.owner == player and w.pos == rc:
                    moves = self.controller.legal_moves_for(w)
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

            clicked_worker = None       #allow switching to another own worker
            for w in self.board.workers:
                if w.owner == player and w.pos == rc:
                    clicked_worker = w
                    break
            if clicked_worker is not None:
                #clicked another worker of same player
                moves = self.controller.legal_moves_for(clicked_worker)
                if moves:
                    self.selected_worker = clicked_worker
                    self.src = rc
                    self.legal = moves
                    self.phase = "select_dst"
                    self.draw(f"{player}: select move for {clicked_worker.id}")
                else:
                    self.draw(f"{player}: no moves for worker ")
                return

            if rc in self.legal and self.selected_worker is not None:
                src = self.src
                dst = rc
                ok,won= self.controller.apply_move(self.selected_worker, dst)
                if not ok:
                    self.draw(f"{player}: illegal move")
                    return

                self.legal = []
                self.phase = "select_build"
                self.draw(f"{player}: moved {src} -> {dst}")

                if won:# win checked
                    self.phase = "game_over"
                    self.draw(f"{player} wins by moving {self.selected_worker.id} to {coords_to_notation(dst)}!")
                    return

                self.legal = self.controller.legal_builds_for(self.selected_worker)
                self.draw(f"{player}: select a build square")

            else:
                self.draw(f"{player}: choose the highlighted cell")
            return

        # click legal build then end turn
        if self.phase == "select_build":
            if rc in self.legal and self.selected_worker is not None:

                ok = self.controller.apply_build(self.selected_worker, rc)
                if not ok:
                    self.draw(f"{player}: illegal build")
                    return

                print(f"[DEBUG] {player} built at {rc}, about to end turn")
                self.controller.end_turn()

                # end turn clear selection
                #
                self.legal =[]
                self.selected_worker = None
                self.src = None
                self.phase = "select_Worker"

                # switch player
                who = self.board.current_player
                self.draw(f"Built at {coords_to_notation(rc)} - {who}'s turn")

                if self.controller.is_ai_turn() and not self.game_over:
                    self.after(50, self.ai_pump)
            else:
                self.draw(f"{player}: choose the highlighted cell")
            return

    def ai_pump(self):
        if self.game_over:
            return

        player = self.board.current_player
        agent = self.controller.players[player]["agent"]
        if agent is None:
            return

        worker, move, build = agent.decide_action(self.board)

        if worker is None or move is None or build is None:
            self.game_over = True
            return

        ok_move, won = self.controller.apply_move(worker, move)
        if not ok_move:
            self.draw(f"{worker.owner}: illegal move by AI")
            self.game_over = True
            return

        ok_build = self.controller.apply_build(worker, build)
        if not ok_build:
            self.draw(f"{worker.owner}: illegal build by AI")
            self.game_over = True
            return

        print(f"[DEBUG] AI {worker.owner} built at {build}, about to end turn")
        self.controller.end_turn()

        self.draw()

        if won:
            self.game_over = True
            self.draw(f"{worker.owner} wins by moving {worker.id} to {coords_to_notation(move)}!")
            return


        if self.controller.is_ai_turn() and not self.game_over:
            self.after(50, self.ai_pump) #continue ai turn after delay

    def any_moves_for(self,player: str) -> bool:        #legal moves for player
        for w in self.board.workers:
            if w.owner == player and self.controller.legal_moves_for(w):
                return True
        return False

    def highlight_selected(self):

        if self.src is None:
            return
        x1, y1, x2, y2 = self._rc_to_xy(*self.src)
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=COLOR_SELECTED, width=4)

    def on_escape(self): #clear selection on escape
        if self.phase in {"select_dst", "select_build"}:
            self.phase = "select_Worker"
            self.selected_worker = None
            self.src = None
            self.legal = []
            self.draw("Selection cleared")