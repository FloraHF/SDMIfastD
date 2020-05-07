import os
import numpy as np 
import pandas as pd
import csv
# import tensorflow as tf
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle

from Games import SDMIGame
from geometries import LineTarget, CircleTarget


fs = 14
# vs = [v_vgreedy, v_mindr, v_ppc, v_ppf]
# dstrs = ['vgreedy', 'mindr', 'ppc', 'ppf']

def compare_value(stras, base, rdir='res30_fixi_10'):
	n = int(rdir.split('_')[-1])
	prefix = '_'.join(rdir.split('_')[:2])
	vs = [np.zeros([n, n]) for _ in stras]
	x = np.zeros([n, n])
	y = np.zeros([n, n])

	for r, d, f in os.walk('results'):
		for v, stra in zip(vs, stras):
			if 'res30_fixi' in r and stra in r and base in r:
				sim = r.split('\\')[1].split('_')
				print('reading from ', r)
				i, j = int(sim[-2]), int(sim[-1])
				# print(n, i, j)
				x[i,j] = 5/n*float(i)
				y[i,j] = 5/n*float(j)
				with open(r+'\\value.csv', 'r') as f:
					for k, line in enumerate(csv.reader(f)):
						# print('getting v')
						if k == 0 and 'value' in line:
							v[i][j] = float(line[-1])

	for v in vs[:]:
		print(v.shape)
		cp = plt.contourf(x, y, v-vs[0])
		plt.colorbar(cp)
		tht = np.linspace(0, 2*np.pi, 50)		
		plt.plot(5 + 1.25*np.cos(tht), 
				 2.5 + 1.25*np.sin(tht), 
				 'k', linewidth=1.5, label='Target')

		plt.gca().tick_params(axis='both', which='major', labelsize=fs)
		plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
		plt.axis("equal")
		plt.axis([0., 4.9, 0., 4.9])
		plt.xlabel('x', fontsize=fs)
		plt.ylabel('y', fontsize=fs)
		plt.axis("equal")

		plt.show()


# compare_value(['vgreedyv', 'vgreedy2v', 'mindr'], 'drx',
# 				rdir='res30_fixi_10')
compare_value(['drx', 'drvp'], 'mindr',
				rdir='res30_fixi_10')