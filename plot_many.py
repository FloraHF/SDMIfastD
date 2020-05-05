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

n = 60
v_mindr = np.zeros([n, n])
v_vgreedy = np.zeros([n, n])
v_ppc = np.zeros([n, n])
v_ppf = np.zeros([n, n])
vs = [v_vgreedy, v_mindr, v_ppc, v_ppf]
dstrs = ['vgreedy', 'mindr', 'ppc', 'ppf']

x = np.zeros([n, n])
y = np.zeros([n, n])

for r, d, f in os.walk('results'):
	for v, dstr in zip(vs, dstrs):
		if 'res30_fixi' in r and dstr in r:
			sim = r.split('\\')[1].split('_')
			n, i, j = int(sim[-3]), int(sim[-2]), int(sim[-1])
			# print(n, i, j)
			x[i,j] = 5/n*float(i)
			y[i,j] = 5/n*float(j)
			with open(r+'\\value.csv', 'r') as f:
				for k, line in enumerate(csv.reader(f)):
					# print('getting v')
					if k == 0 and 'value' in line:
						v[i][j] = float(line[-1])

for v in vs[:]:
	cp = plt.contour(x, y, v)
	# plt.colorbar(cp)
	tht = np.linspace(0, 2*np.pi, 50)		
	plt.plot(5 + 1.25*np.cos(tht), 
			 2.5 + 1.25*np.sin(tht), 
			 'k', linewidth=1.5, label='Target')

	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	# plt.axis("equal")
	plt.axis([0., 4.9, 0., 4.9])
	plt.xlabel('x', fontsize=fs)
	plt.ylabel('y', fontsize=fs)

	plt.show()




