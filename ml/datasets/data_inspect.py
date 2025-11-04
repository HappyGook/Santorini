import numpy as np

def inspect_dataset(path, n=5):
    data = np.load(path)
    keys = list(data.keys())
    print(f"Keys: {keys}")

    # Print shapes for each key
    for key in keys:
        print(f"{key} shape: {data[key].shape}")

    # Determine total samples (assuming first dimension)
    total_samples = None
    for key in keys:
        if len(data[key].shape) > 0:
            total_samples = data[key].shape[0]
            break
    if total_samples is not None:
        print(f"Total samples: {total_samples}")
    else:
        print("Total samples: Unknown")

    # If 'values' exist, print min, max, mean
    if 'values' in data:
        values = data['values']
        print(f"Values stats - min: {values.min():.6f}, max: {values.max():.6f}, mean: {values.mean():.6f}")
    else:
        print("No 'values' key found.")

    print("\nSample data:")
    for i in range(min(n, total_samples if total_samples is not None else 0)):
        print(f"Sample {i}:")
        # Show state summary if exists
        if 'states' in data:
            state = data['states'][i]
            if state.ndim == 1:
                length = state.shape[0]
                root = int(np.sqrt(length))
                if root * root == length:
                    state_reshaped = state.reshape(root, root, -1) if state.size % (root*root) == 0 else state.reshape(root, root)
                else:
                    state_reshaped = None
            elif state.ndim == 2 or state.ndim == 3:
                state_reshaped = state
            else:
                state_reshaped = None

            if state_reshaped is not None:
                # Attempt to detect agent positions from last 3 channels if possible
                if state_reshaped.ndim == 3 and state_reshaped.shape[2] >= 3:
                    player1_channel = state_reshaped[:, :, -3]
                    player2_channel = state_reshaped[:, :, -2]
                    player3_channel = state_reshaped[:, :, -1]

                    p1_positions = np.argwhere(player1_channel > 0)
                    p2_positions = np.argwhere(player2_channel > 0)
                    p3_positions = np.argwhere(player3_channel > 0)

                    p1_pos_str = ', '.join([f"({r},{c})" for r, c in p1_positions]) if p1_positions.size > 0 else "None"
                    p2_pos_str = ', '.join([f"({r},{c})" for r, c in p2_positions]) if p2_positions.size > 0 else "None"
                    p3_pos_str = ', '.join([f"({r},{c})" for r, c in p3_positions]) if p3_positions.size > 0 else "None"

                    print(f"  state: shape={state_reshaped.shape}")
                    print(f"    Player 1 positions: {p1_pos_str}")
                    print(f"    Player 2 positions: {p2_pos_str}")
                    print(f"    Player 3 positions: {p3_pos_str}")
                    # Note: Missing positions may indicate a worker placement or encoding issue.
                elif state_reshaped.ndim == 2:
                    # If 2D, just print shape and basic stats
                    print(f"  state: shape={state_reshaped.shape}, min={state_reshaped.min():.4f}, max={state_reshaped.max():.4f}")
                else:
                    print(f"  state: shape={state_reshaped.shape}")
            else:
                print(f"  state: shape={state.shape} (unable to reshape or interpret)")
        else:
            print("  No 'states' key found.")

        # Show action summary if exists
        if 'actions' in data:
            action = data['actions'][i]
            # Detect action coordinates where action > 0
            if isinstance(action, np.ndarray):
                action_coords = np.argwhere(action > 0)
                if action_coords.size > 0:
                    if action.ndim == 3:
                        coords_str = ', '.join([f"(ch={ch}, r={r}, c={c})" for ch, r, c in action_coords])
                    elif action.ndim == 2:
                        coords_str = ', '.join([f"({r},{c})" for r, c in action_coords])
                    else:
                        coords_str = str(action_coords)
                    print(f"  action: positions with action > 0: {coords_str}")
                else:
                    print("  action: no positions with action > 0")
            else:
                print(f"  action: {action}")
        else:
            print("  No 'actions' key found.")

        # Show value if exists
        if 'values' in data:
            print(f"  value: {data['values'][i]:.6f}")
        else:
            print("  No 'values' key found.")
        print()


inspect_dataset("dataset.npz")