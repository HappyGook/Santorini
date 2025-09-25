from santorini.models import Cell, Worker
from santorini.board import Board
from santorini.rules import can_move, can_build, is_win_after_move, legal_moves, legal_builds

if __name__ == "__main__":
    b = Board()

    # place two workers
    b.place_worker("P1A", (0, 0))
    b.place_worker("P2A", (4, 4))

    # neighbors around (0,0)
    print("Neighbors of (0,0):", b.neighbors((0, 0)))

    # build a step and a dome
    b.build_at((0,0))
    b.build_at((1, 1))  # height 2
    for _ in range(4):  # make a dome at (1,2)
        try:
            b.build_at((1, 2))
        except ValueError:
            pass

    # move P1A one step
    print("Moves from (0,0):", legal_moves(b, (1,1)))   # should include (1,1)
    print("Can move to (1,1)?", can_move(b, (0,0), (1,1)))

    b.move_worker((0, 0), (1, 1))
    b.move_worker((1,1),(2, 2))  # just to demo remove (no worker there)
    print("\nBoard:")
    print("\n".join(b.as_lines(cell_width=6)))


# Simulate win check idea:
# raise (1,1) to level 3, then check moving onto it
    for _ in range(3): b.build_at((1,1))
    print("Win if moving onto (1,1)?", is_win_after_move(b, (0,0), (1,1)))


    print("Builds from (1,1):", legal_builds(b, (1,1)))