import numpy as np

'''
Encode board and action into a tensor for ML input
Flag structure:
0 - height 1
1 - height 2
2 - height 3
3 - dome
4 - 5 - P1 workers
6 - 7 - P2 workers
8 - 9 - P3 workers
10 - Active player
Height 0 is skipped, because if all height flags are 0, then deduce
'''

def encode_board(board, active_player_id):
    tensor = np.zeros((11, 5, 5), dtype=np.float32)
    for r in range(5):
        for c in range(5):
            cell = board.grid[(r, c)]
            h = cell.height
            if h!=0:
                tensor[h-1, r, c] = 1.0 # no flag for height 0, otherwise - 1
            if cell.worker_id:
                w = board.get_worker(cell.worker_id)
                ch = 0
                if w.owner == "P1":
                    ch = 4 if w.id.endswith("A") else 5
                elif w.owner == "P2":
                    ch = 6 if w.id.endswith("A") else 7
                elif w.owner == "P3":
                    ch = 8 if w.id.endswith("A") else 9
                tensor[ch, r, c] = 1.0

                # active player flag
                if w.owner == active_player_id:
                    tensor[10, r, c] = 1.0

    return tensor

def encode_action(worker, move, build):
    tensor = np.zeros((3, 5, 5), dtype=np.float32)
    tensor[0, worker.pos[0], worker.pos[1]] = 1.0
    tensor[1, move[0], move[1]] = 1.0
    tensor[2, build[0], build[1]] = 1.0
    return tensor

def make_input_tensor(board, active_player_id, action):
    print({pid: len(ws) for pid, ws in board.workers.items()})
    board_t = encode_board(board, active_player_id)
    action_t = encode_action(*action)
    return np.concatenate([board_t, action_t], axis=0)  # (14,5,5)