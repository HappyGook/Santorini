import os

import torch
from ai.heuristics import evaluate
from game import rules
from game.board import Board
from ai.agent import Agent
from game.rules import all_legal_actions
from ml.encode import make_input_tensor
from ml.model import SantoNeuroNet, value_loss
from ml.dataset import SantoDataset
from game.moves import place_worker

def make_agents(game_config, model, training_mode):
    if training_mode == "guided":
        return [Agent(f"P{i+1}", algo="maxn", depth=2) for i in range(game_config.num_players)]
    return [Agent(f"P{i+1}", algo="ml", model=model) for i in range(game_config.num_players)]



def selfplay(controller_class, game_config, model_path, dataset_path, num_games=1000, training_mode="selfplay"):
    ml_model = SantoNeuroNet()
    ml_model.load_checkpoint(model_path)
    optimizer = torch.optim.Adam(ml_model.parameters(), lr=1e-4)

    dataset = SantoDataset.load(dataset_path) if os.path.exists(dataset_path) else SantoDataset()
    # Training progress tracking
    training_log = []

    for g in range(num_games):
        board = Board(game_config)

        agents = make_agents(game_config, ml_model, training_mode)

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
            print("\n" + "=" * 60)
            print(f"[TURN {turn_count} BEGIN] Current player: {board.current_player}")
            print("Remaining players:", board.remaining_players)
            print("[WORKERS POSITIONS]")
            for w in board.workers:
                print(f"  {w.id}({w.owner}) at {w.pos} h={board.get_cell(w.pos).height}")
            print("=" * 60)

            pid = board.current_player
            agent = next(a for a in agents if a.player_id == pid)

            if pid not in board.active_players:
                board.next_turn()
                # Normalize current_player_index if board.next_turn assigned a player_id string
                if isinstance(board.current_player_index, str):
                    board.current_player_index = game_config.get_player_index(board.current_player_index)
                continue

            actions = all_legal_actions(board, pid)
            if not actions:
                board.eliminate_player(pid)
                if len(board.active_players) == 1:
                    winner = board.active_players[0]
                    game_over = True
                    break
                board.next_turn()
                # Normalize current_player_index if board.next_turn assigned a player_id string
                if isinstance(board.current_player_index, str):
                    board.current_player_index = game_config.get_player_index(board.current_player_index)
                continue

            ml_score, action = agent.decide_action(board)
            if training_mode == "selfplay":
                game_ml_scores.append(float(ml_score))
            if action is None:
                print(f"[DEBUG] {pid} returned no action.")
                board.next_turn()
                # Normalize current_player_index if board.next_turn assigned a player_id string
                if isinstance(board.current_player_index, str):
                    board.current_player_index = game_config.get_player_index(board.current_player_index)
                turn_count += 1
                continue

            worker, move, build = action

            # Pre-move legality checks
            if not rules.can_move(board, worker.pos, move):
                print(f"❌ Illegal move proposed by {pid}: {worker.pos} -> {move}")
                board.next_turn()
                # Normalize current_player_index if board.next_turn assigned a player_id string
                if isinstance(board.current_player_index, str):
                    board.current_player_index = game_config.get_player_index(board.current_player_index)
                turn_count += 1
                continue
            if not rules.can_build(board, move, build):
                print(f"❌ Illegal build proposed by {pid} at {build}")
                board.next_turn()
                # Normalize current_player_index if board.next_turn assigned a player_id string
                if isinstance(board.current_player_index, str):
                    board.current_player_index = game_config.get_player_index(board.current_player_index)
                turn_count += 1
                continue

            print(f"[TURN {turn_count}] {pid} plays {action}.")

            ok_move, won = controller.apply_move(worker, move)
            if not ok_move:
                print(f"❌ Controller rejected move {move} by {pid}")
                board.next_turn()
                # Normalize current_player_index if board.next_turn assigned a player_id string
                if isinstance(board.current_player_index, str):
                    board.current_player_index = game_config.get_player_index(board.current_player_index)
                turn_count += 1
                continue

            controller.apply_build(worker, build)

            # Grid consistency checks
            cell = board.get_cell(worker.pos)
            if cell.worker_id != worker.id:
                print(f"❌ GRID MISMATCH: worker {worker.id} at {worker.pos} but cell has {cell.worker_id}")

            seen = {}
            for ww in board.workers:
                if ww.pos in seen:
                    print(f"❌ DUPLICATE POSITION: {seen[ww.pos]} and {ww.id} on {ww.pos}")
                seen[ww.pos] = ww.id

            heuristic_score = evaluate(board, pid)
            print(f"[SELFPLAY DEBUG] Score to be saved: {heuristic_score}")

            dataset.add_sample(board, pid, (worker, move, build), float(heuristic_score))
            game_records.append((board.clone(), action, pid, float(heuristic_score)))

            # Win check
            if won or rules.game_over(board):
                print("[WIN CHECK]")
                for r in range(5):
                    for c in range(5):
                        cell = board.grid[(r, c)]
                        if cell.height == 3 and cell.worker_id:
                            print(f"  Worker {cell.worker_id} stands on a win-tile at {(r, c)}")
                winner = pid
                print(f"[GAME OVER] {winner} wins.")
                game_over = True
                break

            board.next_turn()
            # Normalize current_player_index if board.next_turn assigned a player_id string
            if isinstance(board.current_player_index, str):
                board.current_player_index = game_config.get_player_index(board.current_player_index)
            turn_count += 1

        # --- Serialize final board state before saving dataset and metrics ---
        # Use consistent (row, col) ordering: Board uses (row, col) everywhere.
        final_board = []
        for r in range(5):
            row = []
            for c in range(5):
                cell = board.grid[(r, c)]
                worker = next((w.id for w in board.workers if w.pos == (r, c)), None)
                row.append(f"{cell.height}{worker or '.'}")
            final_board.append(' '.join(row))
        final_board_text = '\n'.join(final_board)

        # Training metrics for this game
        game_metrics = {
            'game': g,
            'winner': winner,
            'turns': turn_count,
            'num_moves': len(game_records),
            'final_board_state': final_board_text
        }

        dataset.save(dataset_path)

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