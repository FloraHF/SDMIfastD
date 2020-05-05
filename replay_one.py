import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("ni", help="the number of intruders", type=int)
parser.add_argument("dt", help="time step for simulation", type=float, default=0.01)
args = parser.parse_args()

ni = args.ni
dt = args.dt
xd = np.array([3., 1])

if ni == 1:
	xis=[np.array([0., 5.])]
elif ni == 2:
	xis=[np.array([0., 3.]), np.array([1., 5.])]
elif ni == 3:
	xis=[np.array([0., 2.]),  np.array([0., 4.]), np.array([1., 5.])]

game = SDMIGame(CircleTarget(1.25), ni=ni, dt=dt, res_dir='test_res'+str(ni)+'0')
vgreedyx = DstrategyVGreedy(game, mode='x')
vgreedyv = DstrategyVGreedy(game, mode='v')
vgreedy2x = DstrategyVGreedy2(game, mode='x')
vgreedy2v = DstrategyVGreedy2(game, mode='v')
mindr = DstrategyMinDR(game)
ppc = DstrategyPPC(game)
ppf = DstrategyPPF(game)

drx = IstrategyDRX(game)
drvp = IstrategyDRV(game, mode='pred')
drvg = IstrategyDRV(game, mode='greedy')
dt = IstrategyDT(game)


if ni == 1:
	dstrs = [mindr, ppc, ppf]
	istrs = [drx, drvg, dt]
else:
	dstrs = [vgreedyx, vgreedyv, vgreedy2x, vgreedy2v, mindr, ppc, ppf]
	istrs = [drx, drvp, drvg, dt]

for dstr in dstrs:
	for istr in istrs:

		if str(dstr) == 'mindr' or str(dstr) == 'ppfar': 
			dstr.reset()

		print('New game: '+ str(dstr) + ' vs ' + str(istr))
		game.reset(xd=xd, xis=xis, actives=[1, 1, 1])
		game.advance(16., dstr, istr, draw=False)


