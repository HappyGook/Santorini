"""
This is a symmetry test I made with board transformations to check
if my encoding was correct. All tests were passed with current encoding
but I spent so much time on it that I decided to keep it.
                                                      Oleg
"""

from encode import make_input_tensor
from game.board import Board
import torch
from game.config import GameConfig
from game.models import Worker, BOARD_SIZE
import copy

config = GameConfig(num_players=3)


def horizontal_flip_board(board):
    new_board = board.clone()

    new_grid = {}
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            old_pos = (r, c)
            new_pos = (r, BOARD_SIZE - 1 - c)
            new_cell = copy.copy(new_board.grid[old_pos])
            new_cell.worker_id = None
            new_grid[new_pos] = new_cell

    new_board.grid = new_grid

    for i, worker in enumerate(board.workers):
        pos = worker.pos
        if pos is not None:
            new_pos = (pos[0], BOARD_SIZE - 1 - pos[1])
            new_board.workers[i].pos = new_pos
            new_board.grid[new_pos].worker_id = worker.id

    return new_board


def vertical_flip_board(board):
    new_board = board.clone()

    new_grid = {}
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            old_pos = (r, c)
            new_pos = (BOARD_SIZE - 1 - r, c)
            new_cell = copy.copy(new_board.grid[old_pos])
            new_cell.worker_id = None
            new_grid[new_pos] = new_cell

    new_board.grid = new_grid

    for i, worker in enumerate(board.workers):
        pos = worker.pos
        if pos is not None:
            new_pos = (BOARD_SIZE - 1 - pos[0], pos[1])
            new_board.workers[i].pos = new_pos
            new_board.grid[new_pos].worker_id = worker.id

    return new_board


def rotate_180_board(board):
    new_board = board.clone()

    new_grid = {}
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            old_pos = (r, c)
            new_pos = (BOARD_SIZE - 1 - r, BOARD_SIZE - 1 - c)
            new_cell = copy.copy(new_board.grid[old_pos])
            new_cell.worker_id = None
            new_grid[new_pos] = new_cell

    new_board.grid = new_grid

    for i, worker in enumerate(board.workers):
        pos = worker.pos
        if pos is not None:
            new_pos = (BOARD_SIZE - 1 - pos[0], BOARD_SIZE - 1 - pos[1])
            new_board.workers[i].pos = new_pos
            new_board.grid[new_pos].worker_id = worker.id

    return new_board


def transform_tensor(tensor, transform):
    if transform == "horizontal_flip":
        return torch.flip(tensor, dims=[2])
    elif transform == "vertical_flip":
        return torch.flip(tensor, dims=[1])
    elif transform == "rotate_180":
        return torch.rot90(tensor, 2, dims=[1, 2])
    else:
        return tensor


def transform_action(action, transform, transformed_board):
    """Transform action coordinates AND get the corresponding worker from transformed board"""
    worker, move_pos, build_pos = action

    transformed_worker = transformed_board.get_worker(worker.id)

    if transform == "horizontal_flip":
        new_move = (move_pos[0], BOARD_SIZE - 1 - move_pos[1])
        new_build = (build_pos[0], BOARD_SIZE - 1 - build_pos[1])
    elif transform == "vertical_flip":
        new_move = (BOARD_SIZE - 1 - move_pos[0], move_pos[1])
        new_build = (BOARD_SIZE - 1 - build_pos[0], build_pos[1])
    elif transform == "rotate_180":
        new_move = (BOARD_SIZE - 1 - move_pos[0], BOARD_SIZE - 1 - move_pos[1])
        new_build = (BOARD_SIZE - 1 - build_pos[0], BOARD_SIZE - 1 - build_pos[1])
    else:
        new_move = move_pos
        new_build = build_pos

    return (transformed_worker, new_move, new_build)


def test_board_symmetry():
    board = Board(game_config=config)

    board.workers = [
        Worker("P1A", "P1", (1, 2)),
        Worker("P1B", "P1", (2, 3)),
        Worker("P2A", "P2", (3, 1)),
        Worker("P2B", "P2", (0, 0))
    ]

    board.grid[(1, 2)].height = 1
    board.grid[(1, 2)].worker_id = "P1A"

    board.grid[(2, 3)].height = 2
    board.grid[(2, 3)].worker_id = "P1B"

    board.grid[(3, 1)].height = 3
    board.grid[(3, 1)].worker_id = "P2A"

    board.grid[(0, 0)].height = 1
    board.grid[(0, 0)].worker_id = "P2B"

    player_id = "P1"
    dummy_action = (board.workers[0], (0, 1), (1, 0))  # (worker, move_pos, build_pos)
    original_tensor = make_input_tensor(board, player_id, dummy_action)
    original_tensor = torch.tensor(original_tensor, dtype=torch.float32)

    print("=" * 60)
    print("SYMMETRY TEST")
    print("=" * 60)
    print(f"\nOriginal board - Worker positions:")
    for w in board.workers:
        print(f"  {w.id}: {w.pos}")
    print(
        f"Original action: worker={dummy_action[0].id} at {dummy_action[0].pos}, move={dummy_action[1]}, build={dummy_action[2]}")

    # Horizontal flip
    print("\n" + "-" * 60)
    print("HORIZONTAL FLIP TEST")
    print("-" * 60)
    hflip_board = horizontal_flip_board(board)
    print("Flipped board - Worker positions:")
    for w in hflip_board.workers:
        print(f"  {w.id}: {w.pos}")

    hflip_action = transform_action(dummy_action, "horizontal_flip", hflip_board)
    print(
        f"Transformed action: worker={hflip_action[0].id} at {hflip_action[0].pos}, move={hflip_action[1]}, build={hflip_action[2]}")

    hflip_tensor = torch.tensor(make_input_tensor(hflip_board, player_id, hflip_action), dtype=torch.float32)
    transformed_original = transform_tensor(original_tensor, "horizontal_flip")
    h_match = torch.allclose(hflip_tensor, transformed_original)
    print(f"\n✓ Result: {h_match}")

    if not h_match:
        diff = torch.abs(hflip_tensor - transformed_original)
        print(f"  Max difference: {diff.max().item()}")
        print(f"  Total differences: {(diff > 0).sum().item()}")
        print("  Channels with differences:")
        for ch in range(diff.shape[0]):
            if diff[ch].sum() > 0:
                print(f"    Channel {ch}: {diff[ch].sum().item()} different cells")

    # Vertical flip
    print("\n" + "-" * 60)
    print("VERTICAL FLIP TEST")
    print("-" * 60)
    vflip_board = vertical_flip_board(board)
    print("Flipped board - Worker positions:")
    for w in vflip_board.workers:
        print(f"  {w.id}: {w.pos}")

    vflip_action = transform_action(dummy_action, "vertical_flip", vflip_board)
    print(
        f"Transformed action: worker={vflip_action[0].id} at {vflip_action[0].pos}, move={vflip_action[1]}, build={vflip_action[2]}")

    vflip_tensor = torch.tensor(make_input_tensor(vflip_board, player_id, vflip_action), dtype=torch.float32)
    transformed_original = transform_tensor(original_tensor, "vertical_flip")
    v_match = torch.allclose(vflip_tensor, transformed_original)
    print(f"\n✓ Result: {v_match}")

    if not v_match:
        diff = torch.abs(vflip_tensor - transformed_original)
        print(f"  Max difference: {diff.max().item()}")
        print(f"  Total differences: {(diff > 0).sum().item()}")
        print("  Channels with differences:")
        for ch in range(diff.shape[0]):
            if diff[ch].sum() > 0:
                print(f"    Channel {ch}: {diff[ch].sum().item()} different cells")

    # Rotate 180
    print("\n" + "-" * 60)
    print("ROTATE 180 TEST")
    print("-" * 60)
    rot_board = rotate_180_board(board)
    print("Rotated board - Worker positions:")
    for w in rot_board.workers:
        print(f"  {w.id}: {w.pos}")

    rot_action = transform_action(dummy_action, "rotate_180", rot_board)
    print(
        f"Transformed action: worker={rot_action[0].id} at {rot_action[0].pos}, move={rot_action[1]}, build={rot_action[2]}")

    rot_tensor = torch.tensor(make_input_tensor(rot_board, player_id, rot_action), dtype=torch.float32)
    transformed_original = transform_tensor(original_tensor, "rotate_180")
    r_match = torch.allclose(rot_tensor, transformed_original)
    print(f"\n✓ Result: {r_match}")

    if not r_match:
        diff = torch.abs(rot_tensor - transformed_original)
        print(f"  Max difference: {diff.max().item()}")
        print(f"  Total differences: {(diff > 0).sum().item()}")
        print("  Channels with differences:")
        for ch in range(diff.shape[0]):
            if diff[ch].sum() > 0:
                print(f"    Channel {ch}: {diff[ch].sum().item()} different cells")

    print("\n" + "=" * 60)
    if h_match and v_match and r_match:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("=" * 60)


if __name__ == "__main__":
    test_board_symmetry()