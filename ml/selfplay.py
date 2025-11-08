import os

import torch
from ai.heuristics import evaluate
from game.board import Board
from ai.agent import Agent
from ml.encode import make_input_tensor
from ml.model import SantoNeuroNet, value_loss
from ml.dataset import SantoDataset
from game.moves import place_worker


def selfplay(controller_class, game_config, model_path, dataset_path, num_games=1000, training_mode="selfplay"):
    ml_model = SantoNeuroNet()
    ml_model.load_checkpoint(model_path)
    optimizer = torch.optim.Adam(ml_model.parameters(), lr=1e-4)

    dataset = SantoDataset.load(dataset_path) if os.path.exists(dataset_path) else SantoDataset()

    for g in range(num_games):
        board = Board(game_config)

        # Guided selfplay in maxn to put good moves in dataset
        if training_mode == "guided":
            agents = [Agent(f"P{i+1}", algo="maxn", depth=2) for i in range(game_config.num_players)]
        else:
            agents = [Agent(f"P{i+1}", algo="ml", model=ml_model) for i in range(game_config.num_players)]

        players = {a.player_id: {"type": "AI", "agent": a} for a in agents}
        controller = controller_class(board, players, game_config)

        # place workers
        for agent in agents:
            positions = agent.setup_workers(board)
            for i, pos in enumerate(positions):
                place_worker(board, f"{agent.player_id}{'A' if i == 0 else 'B'}", agent.player_id, pos)
            print(f"{agent.player_id} setup positions: {positions}")



        game_records = []
        winner = None
        game_over = False
        turn_count = 0

        while not game_over and not board.game_over() and turn_count < 500:
            for agent in agents:
                # In guided maxn
                # In selfplay model
                ml_score, action = agent.decide_action(board)
                heuristic_score = evaluate(board, agent.player_id)

                if not action:
                    controller.end_turn()
                    continue

                worker, move, build = action
                ok_move, won = controller.apply_move(worker, move)
                if not ok_move:
                    break
                controller.apply_build(worker, build)

                print(f"[SELFPLAY'S DEBUG] Score to be saved on this turn: {heuristic_score}")
                dataset.add_sample(board, agent.player_id, (worker, move, build), float(heuristic_score))
                game_records.append((board.clone(), action, agent.player_id, float(heuristic_score)))
                controller.end_turn()

                if won or board.game_over():
                    winner = agent.player_id
                    print(f"[GAME OVER] {winner} wins, stopping selfplay loop")
                    game_over = True
                    break

            if winner is not None or game_over:
                break

            print("Workers after turn:", board.workers)
            turn_count += 1

        dataset.save(dataset_path)

        # backprop only in selfplay mode
        if training_mode == "selfplay":
            rewards = {a.player_id: (1.0 if winner == a.player_id else -1.0) for a in agents}

            states, targets = [], []
            gamma = 0.9 # Temporal decay so we reinforce better
            for i, (board, action, pid, heuristic) in enumerate(reversed(game_records)):
                reward = rewards[pid] * (gamma ** i)
                state_tensor = torch.tensor(make_input_tensor(board, pid, action), dtype=torch.float32).unsqueeze(0)
                target_tensor = torch.tensor([reward], dtype=torch.float32)
                states.append(state_tensor)
                targets.append(target_tensor)

            states = torch.cat(states)
            targets = torch.cat(targets)

            ml_model.train()
            optimizer.zero_grad()
            preds = ml_model(states)
            loss_dict = value_loss(preds, targets)
            loss_dict["loss"].backward()
            optimizer.step()

        if g % 10 == 0:
            ml_model.save_checkpoint(model_path, optimizer=optimizer, epoch=g)