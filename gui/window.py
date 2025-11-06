import tkinter as tk
from tkinter import ttk
from typing import Dict, Any

from game.board import Board
from game.config import GameConfig
from game.models import BOARD_SIZE, Worker
from gui.notation import coords_to_notation, GameNotation
from ai.agent import Agent
from gui.gameplay import GameController
from PIL import Image, ImageTk
from pathlib import Path
from game.config import CELL, MARGIN, COLOR_MOVE, COLOR_BUILD, COLOR_SELECTED, PLAYER_COLORS, PLAYER_IMAGES
from game.moves import place_worker



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
    root.geometry("440x420")
    root.resizable(True, True)

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

    ai_box = ttk.LabelFrame(root, text="AI configuration", padding=8)
    ai_box.pack(fill="x", padx=12, pady=(8, 0))
    
    def make_ai_vars():
        return {
            "algo": tk.StringVar(value="minimax"),
            "depth": tk.IntVar(value=3),
            "iters": tk.IntVar(value=400),     # used only for MCTS
        }
    ai_vars: Dict[str, Dict[str, tk.Variable]] = {
        "P1": make_ai_vars(),
        "P2": make_ai_vars(),
        "P3": make_ai_vars(),
    }

     # build a row factory
    rows: Dict[str, Dict[str, Any]] = {}
    def add_ai_row(pid: str, row_index: int):
        r: Dict[str, Any] = {}
        r["frame"] = ttk.Frame(ai_box)
        r["frame"].grid(row=row_index, column=0, sticky="ew", pady=2)

        ttk.Label(r["frame"], text=f"{pid}", width=4).grid(row=0, column=0, padx=(0, 8))
        ttk.Label(r["frame"], text="Algo").grid(row=0, column=1, sticky="e")
        r["algo"] = ttk.Combobox(
            r["frame"], width=10, state="readonly",
            values=["minimax", "maxn", "mcts","rust_mcts"], textvariable=ai_vars[pid]["algo"]
        )
        r["algo"].grid(row=0, column=2, padx=4)

        ttk.Label(r["frame"], text="Depth").grid(row=0, column=3, sticky="e")
        r["depth"] = ttk.Spinbox(r["frame"], from_=1, to=8, width=5, textvariable=ai_vars[pid]["depth"])
        r["depth"].grid(row=0, column=4, padx=4)

        ttk.Label(r["frame"], text="Iters (MCTS)").grid(row=0, column=5, sticky="e")
        r["iters"] = ttk.Spinbox(r["frame"], from_=50, to=2000, increment=50, width=7, textvariable=ai_vars[pid]["iters"])
        r["iters"].grid(row=0, column=6, padx=4)

        rows[pid] = r

    add_ai_row("P1", 0)
    add_ai_row("P2", 1)
    add_ai_row("P3", 2)
    
    def refresh_ai_rows(*_):
        mode = mode_var.get()
        n = int(players_var.get())

          # who is AI by mode
        if mode == "pvp":
            ai_players = set()
        elif mode == "pvai":
            ai_players = {"P2"} if n == 2 else {"P2", "P3"}
        else:  # aivai
            ai_players = {"P1", "P2"} if n == 2 else {"P1", "P2", "P3"}

        # show rows up to n players; enable only AI players
        for i, pid in enumerate(["P1", "P2", "P3"]):
            frame = rows[pid]["frame"]
            if i < n:
                frame.grid()
                is_ai = pid in ai_players
                rows[pid]["algo"].config(state="readonly" if is_ai else "disabled")
                rows[pid]["depth"].config(state="normal" if is_ai else "disabled")
                # toggle iters from algo
                algo = ai_vars[pid]["algo"].get()
                rows[pid]["iters"].config(
                    state="normal" if (is_ai and algo in ("mcts", "rust_mcts")) else "disabled")
            else:
                frame.grid_remove()

    # bind changes
    players_var.trace_add("write", refresh_ai_rows)
    mode_var.trace_add("write", refresh_ai_rows)
    for pid in ["P1", "P2", "P3"]:
        rows[pid]["algo"].bind("<<ComboboxSelected>>", lambda _e, p=pid: refresh_ai_rows())

    refresh_ai_rows()

    selected = {"val": None}

    def start():
        num_players = int(players_var.get())
        mode = mode_var.get()

        # build ai dict only for AI players
        ai: Dict[str, Dict[str, Any]] = {}
        if mode != "pvp":
            if mode == "pvai":
                ai_players = ["P2"] if num_players == 2 else ["P2", "P3"]
            else:  # aivai
                ai_players = ["P1", "P2"] if num_players == 2 else ["P1", "P2", "P3"]
            for pid in ai_players:
                algo = ai_vars[pid]["algo"].get()
                depth = int(ai_vars[pid]["depth"].get())
                iters = int(ai_vars[pid]["iters"].get()) if algo == "mcts" else None
                ai[pid] = {"algo": algo, "depth": depth, "iters": iters}

        selected["val"] = {"num_players": num_players, "mode": mode, "ai": ai}
        root.destroy()

    ttk.Button(root, text="Start Game", command=start, padding=10).pack(pady=20)

    root.mainloop()
    return selected["val"]


def build_players(mode_sel, game_config):
    players = {}
    ids = game_config.player_ids[: mode_sel["num_players"]]
    mode = mode_sel["mode"]
    ai_cfg = mode_sel.get("ai", {})

    players = {}

    for pid in ids:
        if mode == "pvp":
            players[pid] = {"type": "HUMAN"}
        elif mode == "pvai":
            if pid == "P1":
                players[pid] = {"type": "HUMAN"}
            else:
                cfg = ai_cfg.get(pid, {"algo": "minimax", "depth": 3, "iters": None})
                players[pid] = {"type": "AI", "agent": Agent(pid, **cfg)}
        else:  # aivai
            cfg = ai_cfg.get(pid, {"algo": "minimax", "depth": 3, "iters": None})
            players[pid] = {"type": "AI", "agent": Agent(pid, **cfg)}
    return players

class SantoriniTk(tk.Tk):
    def __init__(self, board: Board, controller: GameController, game_config: GameConfig):
        super().__init__()
        self.title("Santorini")
        self.board = board
        self.game_over = False
        self.controller = controller
        self.game_config = game_config
        
        # Initialize game notation
        self.notation = GameNotation()

        w = h = MARGIN * 2 + CELL * BOARD_SIZE

        # Main frame to hold board and dialogue side by side
        main_frame = tk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        # Left: Game board (canvas)
        self.canvas = tk.Canvas(main_frame, width=w, height=h, bg="white")
        self.canvas.pack(side="left", padx=0, pady=0)
        self.load_images()

        # Dialogue and info on the right
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="left", fill="y", padx=10, pady=0)

        # Dialogue area
        dialogue_title = tk.Label(right_frame, text="Dialogue", font=("Arial", 12, "bold"))
        dialogue_title.pack(anchor="nw", pady=(0, 4))
        self.dialogue_label = tk.Label(
            right_frame,
            text="",
            font=("Arial", 11),
            wraplength=220,
            justify="left",
            anchor="nw"
        )
        self.dialogue_label.pack(anchor="nw", fill="x", pady=(0, 8))

        # Status and player info frame
        info_frame = tk.Frame(right_frame)
        info_frame.pack(fill="x")

        self.status = tk.Label(info_frame, text="Board view", anchor="w")
        self.status.pack(fill="x")

        # Player turn indicator
        self.player_info_frame = tk.Frame(info_frame)
        self.player_info_frame.pack(fill="x", pady=5)

        self.create_player_indicators()

        #ui for interaction

        self.phase = "setup"
        self.selected_worker = None
        self.src = None
        self.legal = []
        self.setup_label = "A"

        self.canvas.bind("<Button-1>", self.on_click)
        self.bind("<Escape>", lambda e: self.on_escape())

        self.draw()

        # Start the setup phase immediately
        self.after(50, self.setup_workers)

    def load_images(self):
        #Load & cache all tiles/workers, resized to CELL size
        base = Path(__file__).resolve().parent / "assets"
        tiles_dir = base / "tiles"
        workers_dir = base / "workers"

        tiles ={
            0:"level0.png",
            1:"level1.png",
            2:"level2.png",
            3:"level3.png",
            4:"level4.png"
        }
        self.tiles = {}
        for h, name in tiles.items():
            p = tiles_dir / name
            if p.exists():
                img = Image.open(p).convert("RGBA").resize((CELL, CELL), Image.NEAREST)
                self.tiles[h] = ImageTk.PhotoImage(img)
            else:
                self.tiles[h] = None

        ### worker
        self.worker_img ={}
        for pid in ["P1","P2","P3"]:
            p = base / "workers"/f"{pid}.png"
            if p.exists():
                #worker abit smaller than cell
                w = int(CELL * 0.75)
                img = Image.open(p).convert("RGBA").resize((w, w), Image.LANCZOS)
                self.worker_img[pid] = ImageTk.PhotoImage(img)
            else:
                self.worker_img[pid] = None



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
        placed = place_worker(self.board, wid, pid, pos)

        if placed:
            # Record setup in notation
            worker = next((w for w in self.board.workers if w.id == wid), None)
            if worker:
                self.notation.record_setup(worker)
            self.draw(f"{pid}: placed {wid} at {coords_to_notation(pos)}")
        else:
            self.setup_workers()
            return

        self.setup_workers()


    def cell_empty(self, r:int, c:int) -> bool: #check if cell is empty
        return self.board.grid[(r,c)].worker_id is None

    def setup_workers(self):
        # Determine if all players have placed their workers
        all_players = [self.game_config.get_player_id(i) for i in range(self.game_config.num_players)]
        placed = {pid: sum(1 for w in self.board.workers if w.owner == pid) for pid in all_players}
        all_done = all(placed[pid] == 2 for pid in all_players)

        if all_done:
            self.phase = "select_Worker"
            self.board.current_player_index=0
            self.draw(f"{self.board.current_player}: select worker")
            if self.controller.is_ai_turn() and not self.game_over:
                self.after(50, self.ai_pump)
            return

        # Find next player who still needs to place a worker
        pid = self.board.current_player
        have = placed[pid]
        if have < 2:
            self.setup_label = "A" if have == 0 else "B"
            msg = "place worker A" if have == 0 else "place second worker (B)"
            self.draw(f"{pid}: {msg}")
            if self.controller.is_ai_turn():
                self.after(50, self.ai_setup)   # schedule AI to place ONE worker
            return

        # This player is done, move to next player needing placement
        self.controller.end_turn()
        self.setup_workers()

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

    def cell_box(self, r:int, c:int):
        x1 = MARGIN + c * CELL
        y1 = MARGIN + r * CELL
        x2 = x1 + CELL
        y2 = y1 + CELL
        return x1, y1, x2, y2

    def cell_center(self, r:int, c:int):
        x1, y1, x2, y2 = self.cell_box(r, c)
        return (x1 + x2) // 2, (y1 + y2) // 2

    def draw_tiles(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cell = self.board.grid[(r, c)]
                x1, y1, x2, y2 = self._rc_to_xy(r, c)

                h= max(0, min(4, cell.height))
                img= self.tiles.get(h)
                if img is not None:
                    self.canvas.create_image(x1, y1, image=img, anchor="nw", tags=("tile",))
                else:

                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#ddd", outline="")
    def draw_workers(self):
        for w in self.board.workers:
            r, c = w.pos
            cx,cy = self.cell_center(r,c)

            if self.board.grid[(r,c)].worker_id == w.id:
                sprite = self.worker_img.get(w.owner)
                if sprite:
                    self.canvas.create_image(cx, cy, image=sprite, anchor="center", tags=("worker",))
                else:
                #  text if no PNG exists
                    color = PLAYER_COLORS.get(w.owner, "black")
                    self.canvas.create_text(cx, cy, text=w.id, font=("Arial", 11, "bold"), fill=color)

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
        self.draw_tiles()
        self.draw_workers()

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

    def end_game(self, winner: str):
        """Handle game end - save notation and display winner"""
        self.game_over = True
        self.notation.save()  # Save with default filename
        print(f"Game notation saved! Winner: {winner}")
        self.draw(f"{winner} wins! Game saved.")

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
            ok = place_worker(self.board, wid, player, rc)
            if not ok:
                self.draw(f"{player}: invalid placement")
                return

            # Record setup in notation
            worker = next((w for w in self.board.workers if w.id == wid), None)
            if worker:
                self.notation.record_setup(worker)

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
                    # Record final turn and save notation
                    self.notation.record_turn(src, dst)
                    self.end_game(player)
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

                # Record turn in notation
                self.notation.record_turn(self.src, self.selected_worker.pos, rc)

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

        eval_value, action = agent.decide_action(self.board)
        if action is None:
            # no legal action — treat as stalemate 
            self.controller.end_turn()
            self.draw(f"{player}: no legal moves, skipping")
            if self.controller.is_ai_turn() and not self.game_over:
                self.after(50, self.ai_pump)
            return
        worker, move, build = action

        try:
            player_index = self.game_config.get_player_index(player)
            if isinstance(eval_value, (list, tuple)):
                my_score = float(eval_value[player_index])
            else:
                my_score = float(eval_value)
            phrase = agent.comment_on_eval(max(-1000, min(1000, my_score)))
        except Exception:
            phrase = ""
        # update dialogue
        if hasattr(self, "dialogue_label"):
            self.dialogue_label.config(text=phrase or "")

        old_pos = worker.pos
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

        # Record turn in notation
        if won:
            # Winning move - no build recorded
            self.notation.record_turn(old_pos, move)
        else:
            self.notation.record_turn(old_pos, move, build)

        print(f"[DEBUG] AI {worker.owner} built at {build}, about to end turn")
        self.controller.end_turn()
        self.draw()

        if won:
            self.end_game(worker.owner)
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