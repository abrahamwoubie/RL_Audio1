import random
start_row = 0
start_col = 0
#
goal_row = 2
goal_col = 2
#

wall_row = 2
wall_col = 2

i=0

while (wall_row == start_row and wall_col == start_col):
    wall_row = random.choice(range(0, 2))
    wall_col = random.choice(range(0, 2))

print(wall_row,wall_col)