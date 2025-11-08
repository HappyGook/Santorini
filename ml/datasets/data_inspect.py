import numpy as np
import random


def decode_action(action_tensor):
    """
    Decode action tensor back to readable format.
    Returns: (worker_pos, move_pos, build_pos) or None if no action
    """
    if action_tensor.shape != (3, 5, 5):
        return None

    # Find positions where action > 0 in each channel
    worker_positions = np.argwhere(action_tensor[0] > 0)
    move_positions = np.argwhere(action_tensor[1] > 0)
    build_positions = np.argwhere(action_tensor[2] > 0)

    if len(worker_positions) == 1 and len(move_positions) == 1 and len(build_positions) == 1:
        worker_pos = tuple(worker_positions[0])
        move_pos = tuple(move_positions[0])
        build_pos = tuple(build_positions[0])
        return worker_pos, move_pos, build_pos

    return None


def get_current_player(state_tensor, r, c):
    """Get current player at position (r, c) from the active player flag (channel 10)"""
    if state_tensor[10, r, c] > 0:
        return True
    return False


def render_board_ascii(state_tensor):
    """
    Render 5x5 board state as ASCII with improved spacing.
    Format: height + worker info (e.g., "3", "0", "1 P1A")
    Returns: (board_ascii, current_worker_info) tuple
    """
    board_lines = []
    current_worker_info = None

    for r in range(5):
        row_str = "|"
        for c in range(5):
            # Get height (channels 0-3 for heights 1-4, dome implied by channel 3)
            height = 0
            for h in range(1, 4):  # heights 1, 2, 3
                if state_tensor[h - 1, r, c] > 0:
                    height = h
            if state_tensor[3, r, c] > 0:  # dome
                height = 4

            # Get worker info
            worker_info = ""
            is_current_player = get_current_player(state_tensor, r, c)

            # Check each player's workers
            for player, channels in [("P1", [4, 5]), ("P2", [6, 7]), ("P3", [8, 9])]:
                for i, ch in enumerate(channels):
                    if state_tensor[ch, r, c] > 0:
                        worker_id = f"{player}{'A' if i == 0 else 'B'}"
                        worker_info = f" {worker_id}"
                        if is_current_player:
                            current_worker_info = f"Current worker: {worker_id} at position ({r},{c})"
                        break
                if worker_info:
                    break

            cell_str = f" {height}{worker_info}".ljust(10)  # Add padding for better spacing
            row_str += cell_str

        board_lines.append(row_str)

    return "\n".join(board_lines), current_worker_info


def get_overall_current_player(state_tensor):
    """Determine which player is currently active based on the active player flag"""
    active_positions = np.argwhere(state_tensor[10] > 0)

    if len(active_positions) == 0:
        return "Unknown"

    # Check first active position to determine player
    r, c = active_positions[0]

    # Check which player has a worker at this position
    for player, channels in [("P1", [4, 5]), ("P2", [6, 7]), ("P3", [8, 9])]:
        for ch in channels:
            if state_tensor[ch, r, c] > 0:
                return player

    return "Unknown"


def inspect_dataset(path, n_samples=5, n_visual=4):
    data = np.load(path)
    keys = list(data.keys())
    print(f"Keys: {keys}")

    # Print shapes for each key
    for key in keys:
        print(f"{key} shape: {data[key].shape}")

    # Determine total samples
    total_samples = None
    for key in keys:
        if len(data[key].shape) > 0:
            total_samples = data[key].shape[0]
            break

    if total_samples is not None:
        print(f"Total samples: {total_samples}")
    else:
        print("Total samples: Unknown")
        return

    # Print min, max, mean for values
    if 'values' in data:
        values = data['values']
        print(f"Values stats - min: {values.min():.6f}, max: {values.max():.6f}, mean: {values.mean():.6f}")
    else:
        print("No 'values' key found.")

    print(f"\n{'=' * 80}")
    print(f"VISUAL BOARD INSPECTION - {n_visual} Random Samples")
    print(f"{'=' * 80}")

    # Select random samples for visual inspection
    if total_samples > 0:
        random_indices = random.sample(range(total_samples), min(n_visual, total_samples))

        for i, idx in enumerate(random_indices):
            print(f"\n--- Sample {i + 1} (index {idx}) ---")

            # Get current player
            current_player = "Unknown"
            if 'states' in data:
                state = data['states'][idx]
                current_player = get_overall_current_player(state)

            print(f"Current Player: {current_player}")

            # Render board
            if 'states' in data:
                print("Board State:")
                board_ascii, current_worker_info = render_board_ascii(data['states'][idx])
                print(board_ascii)
                if current_worker_info:
                    print(current_worker_info)

            # Decode and display action
            if 'actions' in data:
                action_info = decode_action(data['actions'][idx])
                if action_info:
                    worker_pos, move_pos, build_pos = action_info
                    print(
                        f"Action: ({worker_pos[0]},{worker_pos[1]})->({move_pos[0]},{move_pos[1]}), ({build_pos[0]},{build_pos[1]})")
                else:
                    print("Action: Unable to decode")

            # Show value
            if 'values' in data:
                print(f"Value: {data['values'][idx]:.6f}")

            print()

    print("\nBasic sample inspection:")
    for i in range(min(n_samples, total_samples if total_samples is not None else 0)):
        print(f"Sample {i}:")

        # Show state summary
        if 'states' in data:
            state = data['states'][i]
            print(f"  state: shape={state.shape}")

            # Show player positions using the encoding channels
            if state.shape == (11, 5, 5):
                for player, channels in [("P1", [4, 5]), ("P2", [6, 7]), ("P3", [8, 9])]:
                    positions = []
                    for j, ch in enumerate(channels):
                        worker_positions = np.argwhere(state[ch] > 0)
                        for pos in worker_positions:
                            worker_id = f"{player}{'A' if j == 0 else 'B'}"
                            positions.append(f"{worker_id}({pos[0]},{pos[1]})")

                    if positions:
                        print(f"    {player} positions: {', '.join(positions)}")
                    else:
                        print(f"    {player} positions: None")

        # Show action summary
        if 'actions' in data:
            action = data['actions'][i]
            action_info = decode_action(action)
            if action_info:
                worker_pos, move_pos, build_pos = action_info
                print(
                    f"  action: ({worker_pos[0]},{worker_pos[1]})->({move_pos[0]},{move_pos[1]}), ({build_pos[0]},{build_pos[1]})")
            else:
                print(f"  action: shape={action.shape} (unable to decode)")

        # Show value
        if 'values' in data:
            print(f"  value: {data['values'][i]:.6f}")
        print()


if __name__ == "__main__":
    inspect_dataset("guided_games.npz")