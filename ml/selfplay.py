import os

import torch
from ai.heuristics import evaluate
from game import rules
from game.board import Board
from ai.agent import Agent
from game.rules import player_has_moves
from ml.encode import make_input_tensor
from ml.model import SantoNeuroNet, value_loss
from ml.dataset import SantoDataset
from game.moves import place_worker

# TODO: Fix the no actions problem (200 moves)
def selfplay(controller_class, game_config, model_path, dataset_path, num_games=1000, training_mode="selfplay"):
    ml_model = SantoNeuroNet()
    ml_model.load_checkpoint(model_path)
    optimizer = torch.optim.Adam(ml_model.parameters(), lr=1e-4)

    dataset = SantoDataset.load(dataset_path) if os.path.exists(dataset_path) else SantoDataset()
    # Training progress tracking
    training_log = []

    for g in range(num_games):
        board = Board(game_config)

        # Guided selfplay in maxn to put good moves in dataset
        if training_mode == "guided":
            agents = [Agent(f"P{i+1}", algo="maxn", depth=2) for i in range(game_config.num_players)]
        else:
            agents = [Agent(f"P{i+1}", algo="ml", model=ml_model) for i in range(game_config.num_players)]

        for agent in agents:
            agent.active = True

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
        game_ml_scores = []  # Track ML scores for the game

        while not game_over and not rules.game_over(board) and turn_count < 200:
            for agent in agents:
                if not agent.active:
                    continue

                if not player_has_moves(board, agent.player_id):
                    agent.active = False
                    print(f"[INFO] {agent.player_id} has no moves and is deactivated at turn {turn_count}")
                    controller.end_turn()
                    continue

                ml_score, action = agent.decide_action(board)
                heuristic_score = evaluate(board, agent.player_id)
                
                # Track ML scores during game
                if ml_score is not None:
                    game_ml_scores.append(ml_score)

                if not action:
                    controller.end_turn()
                    print(f"[DEBUG] {agent.player_id} returned no move at turn {turn_count}")
                    continue
                elif rules.game_over(board):
                    print(f"[TURN {turn_count}] Game over detected.")
                else:
                    print(f"[TURN {turn_count}] {agent.player_id} plays {action}.")

                worker, move, build = action
                ok_move, won = controller.apply_move(worker, move)
                if not ok_move:
                    break
                controller.apply_build(worker, build)

                print(f"[SELFPLAY'S DEBUG] Score to be saved on this turn: {heuristic_score}")
                dataset.add_sample(board, agent.player_id, (worker, move, build), float(heuristic_score))
                game_records.append((board.clone(), action, agent.player_id, float(heuristic_score)))
                controller.end_turn()

                if won or rules.game_over(board):
                    winner = agent.player_id
                    print(f"[GAME OVER] {winner} wins, stopping selfplay loop")
                    game_over = True
                    break

            if winner is not None or game_over:
                break

            print("Workers after turn:", board.workers)
            if rules.game_over(board):
                break
            turn_count += 1

        dataset.save(dataset_path)

        # Training metrics for this game
        game_metrics = {
            'game': g,
            'winner': winner,
            'turns': turn_count,
            'num_moves': len(game_records)
        }

        # Only include ML-related metrics if in selfplay (and if data exists)
        if training_mode == "selfplay" and game_ml_scores:
            game_metrics.update({
                'mean_ml_score': float(sum(game_ml_scores) / len(game_ml_scores)),
                'ml_score_std': float(torch.tensor(game_ml_scores).std().item()) if len(game_ml_scores) > 1 else 0.0,
            })
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
            
            # Compute gradient norm for monitoring
            total_grad_norm = 0.0
            for param in ml_model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            optimizer.step()
            
            # Add training metrics
            game_metrics.update({
                'loss': loss_dict["loss"].item(),
                'mae': loss_dict["mae"].item(),
                'mean_prediction': preds.mean().item(),
                'prediction_std': preds.std().item(),
                'mean_target': targets.mean().item(),
                'target_std': targets.std().item(),
                'grad_norm': total_grad_norm,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        training_log.append(game_metrics)

        if training_mode == "selfplay":
            print(f"[TRAINING PROGRESS] Game {g}: Loss={game_metrics['loss']:.4f}, "
                  f"MAE={game_metrics['mae']:.4f}, Mean Pred={game_metrics['mean_prediction']:.4f}, "
                  f"Grad Norm={game_metrics['grad_norm']:.4f}, Turns={turn_count}")
        else:
            mean_ml = game_metrics.get('mean_ml_score', 0.0)
            print(f"[GAME PROGRESS] Game {g}: Winner={winner}, Turns={turn_count}, Mean ML Score={mean_ml:.4f}")

        if g % 10 == 0:
            ml_model.save_checkpoint(model_path, optimizer=optimizer, epoch=g)
            
            # Print training summary every 10 games
            if training_mode == "selfplay" and len(training_log) >= 10:
                recent_losses = [m['loss'] for m in training_log[-10:] if 'loss' in m]
                recent_maes = [m['mae'] for m in training_log[-10:] if 'mae' in m]
                recent_grad_norms = [m['grad_norm'] for m in training_log[-10:] if 'grad_norm' in m]
                
                if recent_losses:
                    print(f"[SUMMARY] Last 10 games - Avg Loss: {sum(recent_losses)/len(recent_losses):.4f}, "
                          f"Avg MAE: {sum(recent_maes)/len(recent_maes):.4f}, "
                          f"Avg Grad Norm: {sum(recent_grad_norms)/len(recent_grad_norms):.4f}")
    
    # Save final training log
    if training_log:
        import json
        log_path = dataset_path + "_training_log.json"
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"Training log saved to {log_path}")
    
    return training_log