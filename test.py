# import argparse
import os
import numpy as np 
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from Games import SDMIGame
from geometries import LineTarget, CircleTarget
# from strategies.DstrategiesRL import StochRLstrategy, DetermRLstrategy, RLvalue
from strategies.DstrategiesGeo import DstrategyMinDR, DstrategyVGreedy, DstrategyVGreedy2, DstrategyPPF, DstrategyPPC
from strategies.Istrategies import IstrategyDRX, IstrategyDRV, IstrategyDT



# def fractional(n):
# 	print(n)
# 	if n == 1:
# 		return 1
# 	return n*fractional(n-1)


# print(fractional(5))
# parser = argparse.ArgumentParser()
# parser.add_argument("ni", help="the number of intruders", type=int)
# parser.add_argument("dt", help="time step for simulation", type=int)
# args = parser.parse_args()

# ni = 1
# dt = .1
# xd = np.array([3., 1])
# xis=[np.array([0., 5.])]

# ni = 2
# dt = .01
# xd = np.array([3., 1])
# xis=[np.array([0., 3.]), np.array([1., 5.])]

ni = 3
dt = .1
xd = np.array([3., 0])
xis=[np.array([0., 2.]),  np.array([0., 4.]), np.array([2., 5.])]

game = SDMIGame(CircleTarget(1.25), ni=ni, dt=dt, res_dir='res_test_'+str(ni)+'0')

# # game.reset()
# ss = game.get_state()

# print(game.value(ss))
# print(game.value2(ss))


# vgreedy = DstrategyVGreedy(game)
# mindr = DstrategyMinDR(game)
# ppc = DstrategyPPC(game)
# ppf = DstrategyPPF(game)
# # dstrs = [mindr, vgreedy, ppc, ppf]
# dstrs = [ppc]

# drx = IstrategyDRX(game)
# drv = IstrategyDRV(game)
# dt = IstrategyDT(game)
# istrs = [drv]
n = 20

v1, v2 = np.zeros([n, n]), np.zeros([n, n])
xs, ys = np.zeros([n, n]), np.zeros([n, n])

for (i, x) in enumerate(np.linspace(0, 5, n)):
	for (j, y) in enumerate(np.linspace(0, 5, n)):
		print(i, j)
		xd = np.array([x, y])
		game.reset(xd=xd, xis=xis, actives=[1, 1, 1])
		s = game.get_state()
		xs[i, j] = x
		ys[i, j] = y
		v1[i, j] = game.value(s)
		v2[i, j] = game.value2(s)

for v in [v1, v2]:
	cp = plt.contourf(xs, ys, v)
	plt.colorbar(cp)

	plt.gca().tick_params(axis='both', which='major', labelsize=14)
	plt.gca().tick_params(axis='both', which='minor', labelsize=14)
	# plt.axis("equal")
	plt.axis([0., 4.9, 0., 4.9])
	plt.xlabel('x', fontsize=14)
	plt.ylabel('y', fontsize=14)

	plt.show()

# cp = plt.contourf(xs, ys, v1 - v2)
# plt.colorbar(cp)

# plt.gca().tick_params(axis='both', which='major', labelsize=14)
# plt.gca().tick_params(axis='both', which='minor', labelsize=14)
# # plt.axis("equal")
# plt.axis([0., 4.9, 0., 4.9])
# plt.xlabel('x', fontsize=14)
# plt.ylabel('y', fontsize=14)

# plt.show()
for dstr in dstrs:
	for istr in istrs:

		if str(dstr) == 'mindr' or str(dstr) == 'ppfar': 
			dstr.reset()

		print('New game: '+ str(dstr) + ' vs ' + str(istr))
		game.reset(xd=xd, xis=xis, actives=[1, 1, 1])
		game.advance(16., dstr, istr)


