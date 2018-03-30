# Number Shuffle Problem

# Solved State:    Position:
'''
    1 2 3           0 1 2
    4 5 6           3 4 6
    7 8 0           7 8 9
'''

# Solved N State:  Position:
'''
    1  2  3  4     0  1  2  3
    5  6  7  8     4  5  6  7
    9  10 11 12    8  9  10 11
    13 14 15 0     12 13 14 15
'''
# Solved n state:
'''
  1        -   n
  n+1      -   2*n
  ...          ...
  (n-1)n+1 -   n**2
'''
# Position:
'''
   0            - n-1
   n            - 2*n-1
   ...            ...
   (n-1)*n      - n^2
'''
# _ tile represented by n^2
# This  will rep:  1,2,3,4,5,6,7,8,_,...,n^2
solved = np.array(range(n**2))

def get_cost(arr):
  values = np.ones(len(arr))
  total_cost = 0
  for x in range(len(arr)):
    node         =  arr[x]
    cost         =  np.sum(values[0,node])
    values[node] =  0
    total_cost   += cost
  return total_cost

def get_manhattan_distance(arr):



def check_valid_state(arr):
