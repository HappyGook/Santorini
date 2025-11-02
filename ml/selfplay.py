import torch

from ai.heuristics import evaluate
from gui.gameplay import GameController
from game.board import Board
from ai.agent import Agent
from ml.encode import make_input_tensor
from ml.model import SantoNeuroNet, value_loss
from ml.dataset import SantoDataset


def selfplay(controller_class, game_config, num_games: int = 1000):
    ml_model = SantoNeuroNet()
    ml_model.load_checkpoint("learned_models/best.pt")
    optimizer = torch.optim.Adam(ml_model.parameters(), lr=1e-4)
    dataset = SantoDataset.load("datasets/dataset.npz")

    for g in range(num_games):
        board = Board(game_config)
        controller = controller_class(board)

        agents = [Agent(f"P{i+1}", algo="ml", model=ml_model) for i in range(3)]

        # Setup workers properly
        for agent in agents:
            positions = agent.setup_workers(board)
            controller.place_workers(agent.player_id, positions)

        game_records =[]

        winner = None

        # Play game
        while not board.game_over():
            for agent in agents:
                ml_score, action = agent.decide_action(board)
                # Fill dataset with heuristic scores for now, since model hasn't learned it yet
                heuristic_score = evaluate(board, agent.player_id)
                worker, move, build = action

                ok_move, won = controller.apply_move(worker, move)
                if not ok_move:
                    board.print_board()
                    break
                controller.apply_build(worker, build)
                dataset.add_sample(board, agent.player_id, (worker, move, build), float(heuristic_score))
                game_records.append((board.clone(), action, agent.player_id, float(heuristic_score)))
                controller.end_turn()
                if won or board.game_over():
                    winner = agent.player_id
                    break

        # Save dataset once per game
        dataset.save("datasets/dataset.npz")


        # Backpropagation: compute rewards to update model
        rewards = {}

        for agent in agents:
            rewards[agent.player_id] = 1.0 if winner == agent.player_id else -1.0

        states = []
        targets = []
        for board, action, player_id, heuristic in game_records:
            reward = rewards[player_id]
            state_tensor = torch.tensor(make_input_tensor(board,player_id,action), dtype=torch.float32).unsqueeze(0)
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
            ml_model.save_checkpoint("learned_models/best.pt", optimizer=optimizer, epoch=g)