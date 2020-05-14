import argparse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from Games import SDMIGame
from RRTplanner import RRT
from geometries import LineTarget, CircleTarget
from strategies.Istrategies import IstrategyDRX


parser = argparse.ArgumentParser()
parser.add_argument("min_iter", help="minimum iteration", type=int)
parser.add_argument("max_iter", help="maximum iteration", type=int)
args = parser.parse_args()

ni = 3
xd = np.array([3.103, 1.034])

if ni == 1:
	xis=[np.array([0., 5.])]
elif ni == 2:
	xis=[np.array([0., 3.]), np.array([1., 5.])]
elif ni == 3:
	xis=[np.array([0., 2.]),  np.array([0., 4.]), np.array([1., 5.])]

game = SDMIGame(CircleTarget(1.25), ni=ni,
				xd=xd, xis=xis, 
				res_dir='RRTtest_res'+str(ni)+'0_'+str(args.max_iter))
Istrategy = IstrategyDRX(game)

planner = RRT(game, Istrategy,
			  expand_dis=.5,
			  max_iter=args.max_iter, min_iter=args.min_iter, 
			  connect_circle_dist=2.5)

for i in range(20):
	print('####################### trial #',i)
	path, path_i = planner.planning(search_until_max_iter=True, draw=False)
	if path is not None:
		print('!!!!!!!!!!!!!!!!!!!!!!!!!!! found path for trial', i, '!!!!!!!!!!!!!!!!!!!!!!!!!')
		break

# planner.draw_graph()
# plt.plot([x for (x, y) in path], [y for (x, y) in path], '-g')
# circle = Circle(path[0], game.r, color='g', alpha=0.5)
# plt.gca().add_patch(circle)
# for i in range(game.ni):
# 	plt.plot([x for (x, y) in path_i[i]], [y for (x, y) in path_i[i]], '-r')
# plt.grid(True)
# plt.pause(0.01)  # Need for Mac
# plt.show()