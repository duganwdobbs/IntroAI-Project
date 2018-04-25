# Plotter

import matplotlib.pyplot as plt
from math import log
import matplotlib.patches as mpatches

num_sols = [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528]
b_size = range(len(num_sols))
num_sols = [0 if num_sols[x] is 0 else log(num_sols[x]) for x in range(len(num_sols)) ]
plt.plot(b_size,num_sols,'red')
plt.plot(b_size,b_size,'green')
plt.ylabel('e^x Number of Solutions')
plt.xlabel('Board Size')

red_patch   = mpatches.Patch(color='red',   label='Board Solutions')
green_patch = mpatches.Patch(color='green', label='e^x')
plt.legend(handles=[red_patch,green_patch])

plt.show()
