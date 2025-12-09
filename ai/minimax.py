from ai.heuristics import evaluate, order_moves
from game.moves import move_worker, build_block, find_worker_by_id
from game.rules import all_legal_actions
from ml.inference import score_given_actions

# infinity const for evaluation win/lose
INF = 10 ** 9

TT = {}

def ml_order_moves(board, actions, player_index, ml_model, stats):
    if not ml_model or not actions:
        return actions

    player_id = board.game_config.get_player_id(player_index)

    scored = score_given_actions(board, player_id, actions, ml_model, stats)
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

    return [action for action, score in scored_sorted]

def minimax(board, depth, player_index, max_player_index, stats, ml_model=None, maximizing=True):
    stats.bump()
    key = (hash(board), depth, player_index, maximizing)
    if key in TT:
        stats.tt_hits += 1
        return TT[key]

    """
    Alpha-beta minimax strategy
    :param board: current board state
    :param depth: how deep we want to search
    :param player_index: current player as integer (0, 1, 2...)
    :param max_player_index: the player we're maximizing for
    """
    player_id = board.game_config.get_player_id(player_index)
    actions = all_legal_actions(board, player_id)
    if ml_model is not None and actions:
        actions = ml_order_moves(board, actions, player_index, ml_model, stats)
    else:
        actions = order_moves(board, actions)

    if depth == 0 or not actions:
        max_player_id = board.game_config.get_player_id(max_player_index)
        score = evaluate(board, max_player_id)
        TT[key] = (score, None)
        return score, None

    best_action = None
    value = -INF if maximizing else INF

    for action in actions:
        worker, move, build = action
        new_board = board.clone()

        # find worker in new_board
        new_worker = find_worker_by_id(new_board, worker.id)
        if new_worker is None:
            print(f"[DEBUG] Worker id {worker.id} not found in cloned board at depth {depth}")
            continue

        won = move_worker(new_board, new_worker, move)

        if won:
            leaf_score = INF if maximizing else -INF
            TT[key] = (leaf_score, action)  # cache winning child
            return leaf_score, action

        build_block(new_board, new_worker, build)

        # Get next player using modular rotation
        next_player_index = board.game_config.next_player_index(player_index)

        # For 3 players, we need to determine if next player is maximizing
        next_maximizing = (next_player_index == max_player_index)

        score, _ = minimax(new_board, depth - 1, next_player_index, max_player_index, stats,
                           ml_model, next_maximizing)

        # update best action based on maximizing/minimizing
        if maximizing:
            if score > value:
                value = score
                best_action = action

        else:
            if score < value:
                value = score
                best_action = action

    if best_action is None and actions:
        print(f"[WARN] No valid actions selected at depth {depth} for {player_id}, "
              f"actions count={len(actions)}")
        print("Old board:\n")
        board.print_board()
        print("New board:\n")
        new_board.print_board()
        best_action = actions[0]
    TT[key] = (value, best_action)
    return value, best_action


class SearchStats:
    def __init__(self):
        self.nodes = 0
        self.tt_hits = 0
    def bump(self): self.nodes += 1