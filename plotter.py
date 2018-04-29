# Plotter

import matplotlib.pyplot as plt
from math import log
from math import factorial as f
import matplotlib.patches as mpatches

colors = ['b','g','r','c','m','y','k']

num_sols = [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528]
num_sols = [f(x)/2.54**x for x in range(50)]
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

board_times =[[0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0, 2, 2, 3, 3, 5, 5, 6, 9, 10, 11, 11, 13, 14, 14, 15, 16, 17, 18, 18, 19, 20, 23, 24, 25, 26, 29, 33, 34, 37, 39, 42, 44, 51, 74, 82, 88],
              [6, 11, 15, 15, 22, 26, 27, 62, 63, 65, 65, 74, 81, 90, 93, 110, 113, 114, 117, 134, 134, 143, 143, 144, 148, 152, 155, 157, 170, 180, 188, 190, 190, 193, 193, 210, 212, 217, 236, 249, 292, 293, 347, 358, 358, 374, 382, 383, 383, 408, 416, 438, 469, 505, 507, 514, 515, 553, 575, 577, 589, 616, 623, 677, 682, 698, 703, 727, 764, 786, 895, 904, 920, 1055, 1059, 1071, 1099, 1126, 1146, 1193, 1225, 1324, 1385, 1391, 1671, 1801, 1990, 2035, 2201, 2432, 2511, 3420],
              [49, 51, 58, 71, 239, 324, 377, 380, 391, 397, 562, 865, 992, 1194, 1234, 1254, 1270, 1278, 1348, 1449, 1451, 1463, 1533, 1633, 1645, 1648, 1732, 1907, 1950, 1981, 2072, 2101, 2105, 2122, 2140, 2155, 2232, 2232, 2342, 2427, 2447, 2507, 2531, 2608, 2686, 2887, 2897, 2948, 2967, 3000, 3016, 3070, 3129, 3147, 3296, 3303, 3338, 3354, 3429, 3460, 3472, 3563, 3584, 3701, 3783, 3805, 3834, 3836, 3862, 3928, 3976, 4393, 4471, 4603, 4779, 4806, 4834, 4907, 4954, 5076, 5084, 5102, 5128, 5137, 5145, 5162, 5165, 5167, 5204, 5353, 5467, 5469, 5494, 5526, 5840, 5950, 5958, 6126, 6227, 6290, 6340, 6344, 6369, 6480, 6552, 6587, 6597, 6706, 6725, 6816, 6819, 6829, 6831, 6895, 6918, 6966, 7068, 7205, 7215, 7306, 7387, 7481, 7540, 7667, 7673, 7676, 7755, 7796, 7807, 7846, 7971, 8099, 8180, 8271, 8362, 8470, 8489, 8591, 8634, 8754, 8798, 9079, 9088, 9130]
             ]

board_iters =[[1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
              [1, 2, 2, 4],
              [1, 3, 3, 4, 11, 12, 14, 16, 21, 24, 27, 39, 43, 45, 48, 56, 60, 61, 64, 67, 72, 75, 75, 79, 84, 94, 99, 105, 107, 121, 137, 141, 151, 159, 174, 181, 211, 299, 332, 353],
              [21, 39, 52, 53, 73, 88, 91, 202, 206, 211, 213, 241, 266, 294, 304, 358, 369, 373, 382, 435, 438, 467, 468, 469, 483, 497, 506, 512, 555, 588, 616, 620, 621, 631, 631, 689, 694, 709, 769, 812, 945, 949, 1127, 1162, 1162, 1214, 1241, 1244, 1245, 1327, 1351, 1427, 1531, 1644, 1650, 1673, 1676, 1800, 1873, 1880, 1920, 2011, 2035, 2216, 2234, 2286, 2304, 2384, 2504, 2578, 2939, 2970, 3023, 3456, 3469, 3511, 3609, 3702, 3769, 3932, 4033, 4374, 4569, 4591, 5542, 5984, 6635, 6791, 7368, 8159, 8432, 11576],
              [145, 153, 173, 212, 706, 957, 1111, 1122, 1153, 1171, 1635, 2512, 2877, 3456, 3570, 3630, 3675, 3700, 3906, 4199, 4206, 4240, 4450, 4744, 4778, 4785, 5032, 5555, 5683, 5774, 6046, 6131, 6145, 6195, 6249, 6293, 6520, 6520, 6850, 7102, 7163, 7342, 7414, 7636, 7870, 8468, 8498, 8644, 8701, 8797, 8845, 9006, 9179, 9234, 9679, 9699, 9803, 9849, 10073, 10165, 10199, 10472, 10534, 10882, 11126, 11191, 11278, 11284, 11360, 11552, 11693, 12895, 13114, 13490, 14018, 14104, 14194, 14431, 14582, 14974, 14999, 15058, 15140, 15170, 15197, 15250, 15261, 15268, 15387, 15867, 16235, 16241, 16322, 16425, 17430, 17775, 17800, 18320, 18644, 18846, 19003, 19018, 19098, 19454, 19686, 19800, 19830, 20184, 20246, 20538, 20547, 20579, 20587, 20792, 20864, 21019, 21350, 21790, 21823, 22115, 22379, 22680, 22869, 23279, 23296, 23307, 23563, 23695, 23732, 23858, 24260, 24676, 24938, 25233, 25527, 25879, 25939, 26271, 26408, 26780, 26891, 27791, 27815, 27946]
             ]


legends = []
for x in range(len(board_times)):
  size = range(len(board_times[x]))
  times = [[0 if board_times[y][x] is 0 else log(board_times[y][x]) for x in range(len(board_times[y]))] for y in range(len(board_times)) ]
  plt.plot(size,times[x],colors[x])
  legends.append(mpatches.Patch(color=colors[x],   label='Size %d'%(x+4)))

plt.xlabel('Solutions Found')
plt.ylabel('Time Taken')
plt.legend(handles = legends)
plt.show()

legends = []
for x in range(len(board_iters)):
  size = range(len(board_iters[x]))
  plt.plot(size,board_iters[x],colors[x])
  legends.append(mpatches.Patch(color=colors[x],   label='Size %d'%(x+4)))

plt.xlabel('Solutions Found')
plt.ylabel('Iterations Used')
plt.legend(handles = legends)
plt.show()

legends = []
for x in range(len(board_times)):
  size = range(len(board_times[x]))
  plt.plot(board_iters[x],board_times[x],colors[x])
  legends.append(mpatches.Patch(color=colors[x],   label='Size %d'%(x+4)))

plt.xlabel('Iteration')
plt.ylabel('Time')
plt.legend(handles = legends)
plt.show()
