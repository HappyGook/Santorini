import logging
from ml.encode import encode_board, encode_action
from game.rules import all_legal_actions
import torch
logging.basicConfig(level=logging.DEBUG)

def ml_inference(board_state, player_index, model, stats):
    stats.bump()

    player_id = board_state.game_config.get_player_id(player_index)
    legal_actions = all_legal_actions(board_state, player_id)
    if not legal_actions:
        return None, None

    board_tensor = torch.from_numpy(encode_board(board_state, active_player_id=player_id))
    actions_tensor = torch.stack([
        torch.from_numpy(encode_action(*action)) for action in legal_actions
    ])

    values = model.evaluate_actions(board_tensor, actions_tensor)

    best_index = torch.argmax(values).item()
    action = legal_actions[best_index]
    eval_value = values[best_index].item()

    return eval_value, action