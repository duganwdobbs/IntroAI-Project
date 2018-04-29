# Clash Grapher
import matplotlib.pyplot as plt
from math import factorial as f
from math import log
import matplotlib.patches as mpatches

def nCr(n,r):
  return f(n) // (f(r) * f(n-r))


rnge = list(range(3,50))
print(rnge)
ncr_r = [log(nCr(x,2)) for x in rnge]
states = [log(f(x)) for x in rnge]
num_sols = [log(f(x)/2.54**x) for x in rnge]

graphs = [ncr_r,states,num_sols,rnge]
titles = ['Number of Clashes','Number of states','Number of Solutions','e^x']
colors = ['b','g','r','c','m','y','k']
legends = []

for x in range(len(graphs)):
  plt.plot(rnge,graphs[x],colors[x])
  legends.append(mpatches.Patch(color=colors[x],   label=titles[x]))

plt.xlabel('Board Size')
plt.ylabel('e^x Scale')
plt.legend(handles = legends)
plt.show()
