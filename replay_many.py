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
parser.add_argument("lbi", help="directory to read data and save plots", type=int)
parser.add_argument("ubi", help="directory to read data and save plots", type=int)
args = parser.parse_args()

def simulate(xis, dt=0.05, lbi=0, ubi=10, n=20):
	ni = len(xis)
	for i, x in enumerate(np.linspace(0, 5, n)):
		for j, y in enumerate(np.linspace(0, 5, n)):
			if lbi <= i < ubi:
				print('simulation with %d intruders: xd=[%.2f, %.2f], i=%d/%d, j=%d/%d'
						%(ni, x, y, i-lbi+1, ubi-lbi, j+1, n))
				xd = np.array([x, y])
				game = SDMIGame(CircleTarget(1.25), ni=ni, dt=dt, 
								xis=xis, xd=xd,
								res_dir='res'+str(ni)+'0_fixi_%d_%d_%d'%(n, i, j))
				vgreedy = DstrategyVGreedy(game)
				vgreedy2 = DstrategyVGreedy2(game)
				mindr = DstrategyMinDR(game)
				ppc = DstrategyPPC(game)
				ppf = DstrategyPPF(game)

				# for dstr in [vgreedy, vgreedy2, mindr, ppc, ppf]:
				for dstr in [vgreedy, vgreedy2, mindr]:
					game.reset(xd=xd, xis=xis, actives=[1, 1, 1])
					game.advance(2.5, dstr, IstrategyDRV(game, mode='pred'), draw=False, record=False)

# simulate(xis=[np.array([0., 5.])],
# 		dt=.005, n=20,
# 		lbi=0, ubi=20)

# simulate(xis=[np.array([0., 3.]),
# 			 np.array([1., 5.])],
# 		dt=.005, n=50,
# 		lbi=0, ubi=50)

simulate(xis=[np.array([0., 2.]),  
			 np.array([0., 4.]), 
			 np.array([1., 5.])],
		dt=.02, n=10,
		lbi=args.lbi, 
		ubi=args.ubi)





