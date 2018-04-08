# Genetic Tenors
import tensorflow as tf
import numpy as np
import math
import os
# Generate Population
# Generate Cost
# Generate Fitness
# Sort by Fitness
# Selection
# Crossover
# Mutatation
# Repeat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
global num_sols
num_sols = [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528]

global pop_power
pop_power = 10

global board_size
board_size = 8

global pop_size
pop_size = 2**pop_power


def nCr(n,r):
  f = math.factorial
  return f(n) // (f(r) * f(n-r))

def init():
  pop_shape  = [pop_size,board_size]
  population = tf.random_uniform(pop_shape,minval=0,maxval=board_size,dtype = tf.int32)
  return population

def kernel_generation():
  center = board_size -1
  k_size = board_size + center
  k_shape= [int(k_size),int(board_size),1,1]
  kernel = np.zeros(k_shape,dtype=np.uint8)
  for b in range(board_size):
    kernel[center    ,b] = 1
    kernel[center + b,b] = 1
    kernel[center - b,b] = 1
  kernel[center,0] = 0
  return kernel

def get_fitness(population):
  with tf.variable_scope("FitnessGeneration") as scope:
    ohops = tf.one_hot(population,board_size)
    tf.summary.image("Board",tf.reshape(ohops*255,[pop_size,board_size,board_size,1]))
    pops = tf.reshape(ohops,[pop_size,board_size,board_size])
    pad_pops=[]
    for x in range(pop_size):
      pad_pops.append(tf.pad(pops[x],([board_size-1,board_size-1],[0,board_size-1])))
    pops = tf.stack(pad_pops)
    pops = tf.reshape(pops,[pops.shape[0].value,pops.shape[1].value,pops.shape[2].value,1])
    kern = kernel_generation()
    costs = tf.nn.conv2d(pops,kern,[1,1,1,1],padding='VALID')
    costs = tf.squeeze(costs)
    costs = ohops * costs
    # Reduce along x and y to have a batch size length tensor of total costs.
    costs = tf.reduce_sum(costs,-1)
    costs = tf.reduce_sum(costs,-1)
    fitness = nCr(board_size,2) - costs
    return fitness,costs

def get_tops(population,fitness):
  with tf.variable_scope("GetTops") as scope:
    top_fits,inds = tf.nn.top_k(fitness,10,sorted=False)
    # top_pops = tf.gather_nd(population,inds)
    return top_fits,inds

def march_madness(population,fitness):
  with tf.variable_scope("MartchMadness") as scope:
    children = []
    bracket=0
    while population.shape[0].value > 1:
      with tf.variable_scope("Tier_%d"%bracket) as scope:
        bracket+=1
        winners  = []
        print("Tournament size: %d"%population.shape[0].value)
        for x in range(population.shape[0] // 2):
          print("Bracket %d; Creating match %d / %d"%(bracket,x,population.shape[0] // 2))
          child,winner = match(population[x],fitness[x],population[-(x+1)],fitness[-(x+1)])
          children.append(child)
          winners.append(winner)
        population = tf.stack(winners)
    # append the tournament victor to the pool
    children.append(tf.squeeze(population))
    children = tf.stack(children,0)
    return children

def match(pop1,fit1,pop2,fit2):
  with tf.variable_scope("Match") as scope:
    pop1 = tf.squeeze(pop1)
    pop2 = tf.squeeze(pop2)
    fit1 = tf.squeeze(fit1)
    child = tf.squeeze(tf.random_uniform((1,board_size),maxval = board_size,dtype=tf.int32))
    child = tf.where(tf.equal(pop1,pop2),pop1,child)
    approx = tf.squeeze(tf.random_uniform((1,1),dtype = tf.float32))
    winner = tf.where((approx > fit1),pop1,pop2)
    return child,winner

def tf_genetic_algo():
  with tf.Graph().as_default():
    sess          = tf.Session()

    pop_shape  = [pop_size,board_size]
    population = tf.get_variable("population",shape = pop_shape,dtype=tf.int32,initializer=tf.random_uniform_initializer(minval=0,maxval=board_size,dtype=tf.int32))
    # population      = tf.random_uniform(pop_shape,minval=0,maxval=board_size,dtype = tf.int32,name = 'population')
    # population    = tf.placeholder(shape=[pop_size,board_size],name = 'population',dtype=tf.int32)

    print("Creating fitness operation...")
    fitness,costs = get_fitness(population)
    print("Creating top populations operation...")
    top_fits,inds = get_tops(population,fitness)
    print("Creating new population operation... (Selection, crossover, mutation)")
    children      = march_madness(population,fitness)
    population = population.assign(children)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    print("Done initialing graph operations...")

    step      = 0
    solutions = []
    new_pop = np.random.uniform(size = (2**pop_power,board_size),low = 0, high = board_size)
    summaries = tf.summary.merge_all()
    writer    = tf.summary.FileWriter("./",sess.graph)
    print("Initalzing GPU operations...")


    run_op = [population,top_fits,inds,summaries]
    while len(solutions) < num_sols[board_size]:
      step +=1
      print("Iteration %d"%step)
      _,_top_fits,_inds,_summ = sess.run(run_op)
      writer.add_summary(_summ,step)
      for x in range(len(_top_fits)):
        if _top_fits[x] == nCr(board_size,2):
          solutions.append("Some solution?")
          print("Some solution found, total %d"%len(solutions))

tf_genetic_algo()
