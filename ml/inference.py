import logging
logging.basicConfig(level=logging.DEBUG)

def ml_inference(board_state, player_index, model, stats):
    stats.bump()

    from ml.encode import encode_board, encode_action
    from game.rules import all_legal_actions
    import torch

    player_id = board_state.game_config.get_player_id(player_index)
    legal_actions = all_legal_actions(board_state, player_id)
    logging.debug(f"[ML-Inference] Found {len(legal_actions)} legal actions for player {player_index}")
    if not legal_actions:
        return None, None

    logging.debug(f"[ML-Inference] Example action: {legal_actions[0] if legal_actions else 'None'}")

    board_tensor = torch.from_numpy(encode_board(board_state, player_index))
    actions_tensor = torch.stack([
        torch.from_numpy(encode_action(*action)) for action in legal_actions
    ])

    values = model.evaluate_actions(board_tensor, actions_tensor)
    logging.debug(f"[ML-Inference] Evaluated {len(values)} actions: values={values.tolist()}")

    best_index = torch.argmax(values).item()
    action = legal_actions[best_index]
    eval_value = values[best_index].item()

    logging.debug(f"[ML-Inference] Selected action index={best_index}, action={action}, value={eval_value:.4f}")
    return eval_value, action