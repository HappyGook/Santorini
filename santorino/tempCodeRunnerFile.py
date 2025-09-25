from santorini.board import Board

b = Board()

# Setup
b.place_worker("P1A", (0, 0))
b.place_worker("P2A", (4, 4))

# Read helpers
assert b.in_bounds((2,2)) is True
assert (1,1) in b.neighbors((0,0))
assert b.is_occupied((0,0)) is True
assert b.get_cell((0,0)).height == 0

# Safe move (assumes rules said OK)
b.move_worker((0,0), (1,1))
assert b.is_occupied((1,1)) and not b.is_occupied((0,0))

# Build (assumes rules said OK)
b.build_at((2,2))   # height 1
for _ in range(3):
    b.build_at((2,2))  # height 4 = dome
try:
    b.build_at((2,2))  # ❌ cannot build over dome
except ValueError:
    pass
