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

ni = 2
dt = .01
xd = np.array([3., 1])
xis=[np.array([0., 3.]), np.array([1., 5.])]

# ni = 3
# dt = .1
# xd = np.array([3., 0])
# xis=[np.array([0., 2.]),  np.array([0., 4.]), np.array([2., 5.])]

game = SDMIGame(CircleTarget(1.25), ni=ni, dt=dt, res_dir='debug_res'+str(ni)+'0')
ss = game.get_state()
print(game.value(ss))
print(game.value2(ss))






