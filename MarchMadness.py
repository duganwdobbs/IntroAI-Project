import numpy      as np
import random
import math
import datetime
# Flags:
#       Board Size
board_size = 11
#       Population Size
pop_size   = 10000

# Number of solutions for n queens, credit https://oeis.org/A002562
num_sols = [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528]
times = []
iterations = []
global solutions
solutions = []

global tourn_size
tourn_size = int(math.log2(pop_size - 1)) + 2

def nCr(n,r):
  f = math.factorial
  return f(n) // (f(r) * f(n-r))

def init_states(board_size,pop_size):
  population = np.random.uniform(size = (pop_size,board_size),low = 0, high = board_size)
  population = population.astype(np.int32)
  return population

def gen_fitness(start,iter,population):
  population = np.array(population)
  max_clash = (nCr(board_size,2))
  # fitness = # of horiz + diag clashes of this queen to end queen iterative
  clashes = []
  for p in range(pop_size):
    clash = 0
    for x in range(board_size-1):
      q_pos = population[p][x]
      for y in range(x+1,board_size):
        r_pos = population[p][y]
        if ((q_pos==r_pos) or (q_pos == (r_pos + (y-x))) or (q_pos == (r_pos - (y-x)))):
          clash += 1
    clashes.append(clash)
  clashes = np.array(clashes)
  fitness = max_clash - clashes
  for x in range(len(fitness)):
    if fitness[x] == max_clash:
      if not in_solutions(population[x]):
        print("SOLUTION FOUND, %d of %d"%(len(solutions)+1,num_sols[board_size-1]),end=' ')
        delta = datetime.datetime.now() - start
        times.append(delta.seconds)
        iterations.append(iter)
        print(population[x])
        solutions.append(population[x])

      else:
        # Solution is found, fitness is 0. Won't be picked as a parent.
        fitness[x] = 0
  fitness = fitness / np.sum(fitness)
  return fitness

def to_dec(state):
  sum = 0
  for x in range(len(state)):
    sum+=state[x] * 10**x
  return sum

def equals(state1,state2):
  return to_dec(state1) == to_dec(state2)

def in_solutions(state):
  if solutions is None:
    return False
  for solution in solutions:
    if equals(state,solution):
      return True
  return False


def selection_tournament(fitness,k):
  best = None
  for i in range(k):
    ind = None
    while ind is None or fitness[ind] * pop_size < .5:
      ind = random.randint(0,pop_size-1)
    if best is None or fitness[best] < fitness[ind]:
      best = ind
  return best

def crossover_3tournament(inds,population):
  childs = []
  parent1 = population[inds[0]]
  parent2 = population[inds[1]]
  # Note, explore exclusive in the future
  # domain  = list(np.range(0,board_size))
  for x in range(2):
    child = np.random.uniform(size = (board_size),low = 0, high = board_size)
    child = child.astype(np.int32)
    for y in range(board_size):
      if parent1[y] == parent2[y]:
        child[y] = parent1[y]
    childs.append(child)
  temp = []
  for child in childs:
    for single in child:
      temp.append(single)
  return temp

def march_madness(fitness,population):
  global pop_size # We need to change this global, not make a new one.
  pop_size = 2 ** (tourn_size-2)
  # Receives fitness and states, pairs, then sorts in desc
  paired  = [[fitness[x],population[x]] for x in range(len(population))]
  sorting = sorted(paired,key = lambda val: val[0])
  sorting = sorting[::-1]

  tourn_pop = march_selection(sorting)
  children   = march_crossover(tourn_pop)

  return children

def march_selection(sorting):
  # Throwing out the lower amount
  sorting = sorting[0:pop_size]

  # Creating a proper tiered structure, so first and second seed appear on
  # opposite ends
  for x in range(1,len(sorting) // 2):
    sorting[x],sorting[-x] = sorting[-x],sorting[x]
  return sorting

def march_crossover(tourn_pop):
  children = []
  while len(tourn_pop) > 1:
    next_tier = []
    #Tournament Tier
    for x in range(len(tourn_pop)//2):
      new_child,winner = march_tourn(tourn_pop[2*x],tourn_pop[2*x+1])
      if winner == 0:
        next_tier.append(tourn_pop[2*x])
      else:
        next_tier.append(tourn_pop[2*x+1])
      children.append(new_child)
    tourn_pop = next_tier

  # Winner added to new population
  children.append(tourn_pop[0][1])

  return children


def march_tourn(team1,team2):

  fit1, parent1 = team1
  fit2, parent2 = team2
  eps           = fit1 + fit2 + 1e-5
  fit1,fit2 = fit1 / eps,fit2/eps
  winner = 1

  # Generate a random percentage and see if team2 wins intead of team1
  if random.random() > fit1:
    winner = 0

  for x in range(2):
    child = np.random.uniform(size = (board_size),low = 0, high = board_size)
    child = child.astype(np.int32)
    for y in range(board_size):
      if parent1[y] == parent2[y]:
        child[y] = parent1[y]
  return child,winner

def single_mutation(child):
  rand = random.randint(0,100)
  if rand < 10:
    allele1 = random.randint(0,board_size-1)
    allele2 = random.randint(0,board_size-1)
    child[allele1],child[allele2] = child[allele2],child[allele1]
  return child

def double_mutation(child):
  child = single_mutation(child)
  child = single_mutation(child)
  return child

def main():
  population = init_states(board_size,pop_size)
  iter = 0
  try:
    while len(solutions)  < num_sols[board_size-1] :
      iter += 1
      fitness = []
      fitness = gen_fitness(start,iter,population)

      children = march_madness(fitness,population)

      # inds     = [[selection_tournament(fitness,2),selection_tournament(fitness,2)] for x in range(len(fitness) // 2)]
      # children = [crossover_3tournament(ind,population) for ind in inds]

      children = [single_mutation(child) for child in children]
      if iter%100 == 0:
        print("Iteration %d"%iter)
      population = children
  except:
    pass
  [print(solution) for solution in solutions]
  print("Times: ",times)
  print("Iterations: ",iterations)
  print("FOUND IN %d ITERATIONS, %d TOTAL INDIVIDUALS"%(iter,iter * pop_size))

main()
