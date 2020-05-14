import argparse
import numpy as np 
import pandas as pd
import csv
import os
# import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

from Games import SDMIGame
from strategies.Istrategies import IstrategyDRV
from geometries import LineTarget, CircleTarget

parser = argparse.ArgumentParser()
parser.add_argument("res_dir", help="directory to read data and save plots", type=str)
args = parser.parse_args()

rdir = args.res_dir
with open('results/'+rdir +'/param.csv') as f:
    lines = csv.reader(f)
    for data in lines:
    	if 'r' == data[0]:
    		r = float(data[1])
    	if 'vd' == data[0]:
    		vd = float(data[1])
    	if 'vi' == data[0]:
    		vi = float(data[1])
    	if 'ni' == data[0]:
    		ni = int(float(data[1]))
    	if 'target' == data[0]:
    		R = float(data[1].split('_')[-1])

game = SDMIGame(CircleTarget(R), ni=ni,
				vd=vd, vi=vi,
				res_dir=rdir)
drv = IstrategyDRV(game)

fs = 32
fsl = 24
colors = ['green', 'red', 'blue', 'm', 'k']

def read_data(dr):
	print('reading from', dr)
	data = pd.read_csv(dr+'/state.csv')
	t = data.filter(regex='time').to_numpy()
	ss = data.filter(regex='state').to_numpy()
	if 'RRT' in dr:
		ss = np.flip(ss, 0)
		ts = []
		for i in range(len(t)):
			ts.append(sum(t[i:]))
		t = np.flip(np.asarray(ts))
	n = int((len(ss[0]) - 4)/5)

	actives = ss[:,-n:]
	xds = []
	xiss = [[] for _ in range(n)]
	for s in ss:
		xis, xd, _, _, _ = game.unwrap_state(s)
		xds.append(xd)
		for i in range(n):
			xiss[i].append(xis[i])
	xd = np.asarray(xds)			
	for i in range(n):
		xiss[i] = np.asarray(xiss[i])
	
	with open(dr+'/value.csv', 'r') as f:
		for data in csv.reader(f):
			if 'value' == data[0]:
				value = data[1].strip()
	capids = []
	last_ncap = -1
	for i, active in enumerate(actives):
		ncap = n - sum(active)
		# print(active)
		if (ncap == last_ncap+1) and (ncap > last_ncap):
			capids.append(i)
			last_ncap = ncap

	# print(xis)
	return n, t*10, xiss, xd, ss, capids, value

def read_data_concise(dr):
	print('reading from', dr)
	data = pd.read_csv(dr+'/state.csv')
	t = data.filter(regex='time').to_numpy()
	ss = data.filter(regex='state').to_numpy()
	return t, ss

def plot_target():
	tht = np.linspace(.55*np.pi, 1.35*np.pi, 50)		
	plt.plot(game.target.x0 + R*np.cos(tht), 
			 game.target.y0 + R*np.sin(tht), 
			 'k', linewidth=3, label='Target')

def compare_traj(stras, base, trange=-1, RRT_traj=[]):
	fig = plt.figure(figsize=(8,8))
	plot_target()
	print(stras+RRT_traj)
	
	for j, (stra, c) in enumerate(zip(stras+RRT_traj, colors)):
		print(stra, c)
		if isinstance(stra, int):
			n, t, xis, xd, _, capids, value = read_data('results/RRTtest_res30_'+str(stra)+'/RRT')
			stra = 'RRT'+str(stra)
			# print(capids)
		else:
			try: n, t, xis, xd, _, capids, value = read_data('results/'+rdir+'/'+'_'.join([stra, base]))
			except: n, t, xis, xd, _, capids, value = read_data('results/'+rdir+'/'+'_'.join([base, stra]))
			if 'drv' in stra or 'drx' in stra:
				value = '%.2f'%game.target.level(xis[1][-1,:])
		tmax = len(t)-1 if trange == -1 else trange

		# plot trajectories
		plt.plot(xd[:tmax,0], xd[:tmax,1], color=c, linewidth=3, linestyle=(0,()), label=stra)
		for i in range(n):
			plt.plot(xis[i][:tmax,0], xis[i][:tmax,1], color=c, linewidth=3, linestyle=(0, (10,4)))
			if j == 1:
				plt.text(xis[i][0,0], xis[i][0,1]-.5, r'$I_'+str(i)+'$', fontsize=fs)	

		# plot markers
		# print(len(t), len(xis[0]), capids+[tmax])
		for i, k in enumerate(capids):
			if k <= tmax:
				plt.plot(xd[k,0], xd[k,1], color=c, marker='o', markersize=12)
				if i != 0:
					circle = Circle(xd[k], r, color=c, alpha=0.2)
					plt.gca().add_patch(circle)
				for i in range(n):
					plt.plot(xis[i][k,0], xis[i][k,1], color=c, marker='>', markersize=12)
	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.axis("equal")
	plt.axis([0., 5., 0., 5.])
	plt.xlabel(r'$x$', fontsize=fs)
	plt.ylabel(r'$y$', fontsize=fs)
	plt.legend(fontsize=fs-5, loc='lower right', handlelength=.9)
	plt.subplots_adjust(left=0.1, right=0.99, top=0.96, bottom=0.15)
	plt.show()
	# plt.savefig('results/'+rdir+'/traj_'+dstr+'.png')
	# plt.close()

def compare_value(stras, base):
	fig = plt.figure(figsize=(8,3))
	
	for stra, c in zip(stras, colors):
		try: _, t, _, _, ss, _, _ = read_data('results/'+rdir+'/'+'_'.join([stra, base]))
		except: _, t, _, _, ss, _, _ = read_data('results/'+rdir+'/'+'_'.join([base, stra]))

		plt.plot(t, [game.value2(s) for s in ss], linewidth=2, label=stra, color=c)

	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fsl)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fsl)
	plt.xlabel(r'$t$', fontsize=fsl)
	plt.ylabel(r'$V^1$', fontsize=fsl)
	plt.legend(fontsize=fsl, ncol=1, loc='best')
	plt.subplots_adjust(left=0.14, right=0.99, top=0.99, bottom=0.26)
	plt.show()

def plot_icurr(dstr):
	fig = plt.figure(figsize=(8,3))
	
	n, t, _, _, ss, _, _ = read_data('results/'+rdir+'/'+'_'.join([dstr, 'drvp']))

	err = np.array([drv.get_icurr(s) for s in ss[:-1]])
	for i in range(n):
		plt.plot(t[:-1], err[:,i], linewidth=2, label=r'$a_'+str(i)+'$', color=colors[i])

	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fsl)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fsl)
	plt.xlabel(r'$t$', fontsize=fsl)
	plt.ylabel(r'$a$', fontsize=fsl)
	plt.ylim((-0.1, 1.1))
	
	plt.legend(fontsize=fsl, loc='best')
	plt.subplots_adjust(left=0.13, right=0.96, top=0.99, bottom=0.27)
	plt.show()

def compare_order(dstr, istr):
	fig = plt.figure(figsize=(8,3))
	_, t, _, _, ss, _, _ = read_data('results/'+rdir+'/'+'_'.join([dstr, istr]))

	for order, c in zip(game.orders, ['r', 'b', 'k', 'm', 'g', 'c']):
		plt.plot(t, [game.value2_order(s, order) for s in ss], 
				color=c, linestyle='--', linewidth=2, label=','.join(map(str,order)))
	# plt.plot(t, [game.value2(s) for s in ss], color='b', linewidth=2, label='value')	

	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fsl)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fsl)
	plt.xlabel(r'$t$', fontsize=fsl)
	plt.ylabel(r'$\bar{V}^1$', fontsize=fsl)
	plt.legend(fontsize=fsl, ncol=2, loc='best')
	plt.subplots_adjust(left=0.11, right=0.99, top=0.99, bottom=0.26)
	# plt.savefig('results/'+rdir+'/value_'+'_'.join([dstr, istr])+'.png')
	# plt.close()
	plt.show()

def compare_value12(dstr, istr):
	fig = plt.figure(figsize=(8,3))
	_, t, _, _, ss, _, _ = read_data('results/'+rdir+'/'+'_'.join([dstr, istr]))

	# for order, c in zip(game.orders, ['r', 'b', 'k', 'm', 'g', 'c']):
	# 	plt.plot(t, [game.value_order(s, order) for s in ss], color=c, linestyle='--', label=','.join(map(str,order)))
	plt.plot(t, [game.value(s) for s in ss], 'r', linewidth=2, label=r'$V^N$',)	
	plt.plot(t, [game.value2(s) for s in ss], 'g', linewidth=2, label=r'$V^1$')	

	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fsl)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fsl)
	plt.xlabel(r'$t$', fontsize=fsl)
	plt.ylabel(r'Value', fontsize=fsl)
	plt.legend(fontsize=fsl)
	plt.subplots_adjust(left=0.11, right=0.99, top=0.99, bottom=0.26)
	# plt.savefig('results/'+rdir+'/value_'+'_'.join([dstr, istr])+'.png')
	# plt.close()
	plt.show()

def compare_vdcontour(stras, base, rdir='res30_fixi_10'):
	n = int(rdir.split('_')[-1])
	prefix = '_'.join(rdir.split('_')[:2])
	vs = [np.zeros([n, n]) for _ in stras]
	x = np.zeros([n, n])
	y = np.zeros([n, n])
	m = len(stras)

	for r, d, f in os.walk('results'):
		for v, stra in zip(vs, stras):
			if 'res30_fixi' in r and stra in r and base in r:
				sim = r.split('\\')[1].split('_')
				i, j = int(sim[-2]), int(sim[-1])
				x[i,j] = 5/n*float(i)
				y[i,j] = 5/n*float(j)
				with open(r+'\\value.csv', 'r') as f:
					for k, line in enumerate(csv.reader(f)):
						# print('getting v')
						if k == 0 and 'value' in line:
							v[i][j] = float(line[-1])
	for (i, v) in enumerate(vs):
		fig = plt.figure(figsize=(8,8))
		cp = plt.contourf(x, y, v, levels=[-1.5, -1, -.5, 0, .25, .5, .75, 1.5, 1.75, 2, 2.25, 2.5])
		plot_target()
		bar = plt.colorbar(cp)
		bar.ax.tick_params(labelsize=fs) 

		plt.gca().tick_params(axis='both', which='major', labelsize=fs)
		plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
		# plt.axis("equal")
		# plt.axis([0., 4.9, 0., 4.9])
		plt.xlabel(r'$x$', fontsize=fs)
		plt.ylabel(r'$y$', fontsize=fs)
		plt.subplots_adjust(left=0.1, right=0.93, top=0.88, bottom=0.15)

		plt.show()

def compare_vicontour(stras, base, rdir='res30_fixi_10'):
	n = int(rdir.split('_')[-1])
	prefix = '_'.join(rdir.split('_')[:2])
	vs = [np.zeros([n, n]) for _ in stras]
	x = np.zeros([n, n])
	y = np.zeros([n, n])

	for r, d, f in os.walk('results'):
		for v, stra in zip(vs, stras):
			if 'res30_fixi' in r and stra in r and base in r:
				_, ss = read_data_concise(r)
				sim = r.split('\\')[1].split('_')
				i, j = int(sim[-2]), int(sim[-1])
				x[i,j] = 5/n*float(i)
				y[i,j] = 5/n*float(j)
				v[i][j] = game.valuei(ss[-1, :])[1]

	for v in vs[1:2]:
		fig = plt.figure(figsize=(8,8))
		cp = plt.contourf(x, y, vs[-1]-v, levels=[-.06, -.03, 0, .03, .06, .09, .12, .15])
		plot_target()
		bar = plt.colorbar(cp)
		bar.ax.tick_params(labelsize=fs) 

		plt.gca().tick_params(axis='both', which='major', labelsize=fs)
		plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
		# plt.axis("equal")
		# plt.axis([0., 4.9, 0., 4.9])
		plt.xlabel(r'$x$', fontsize=fs)
		plt.ylabel(r'$y$', fontsize=fs)
		plt.subplots_adjust(left=0.1, right=0.93, top=0.88, bottom=0.15)

		plt.show()

######## Fig.4 ########
# compare_traj(['vgreedy2v'], 'drx', RRT_traj=[1200, 25000])
# compare_order('vgreedy2v', 'drx')
# compare_value12('vgreedy2v', 'drx')

######## Fig.5 ########
# compare_traj(['vgreedy2v', 'vgreedyv', 'mindr'], 'drx')
# compare_value(['vgreedy2v', 'vgreedyv', 'mindr'], 'drvp')
# plot_icurr('vgreedyv')
######## Fig.6 ########
# compare_vdcontour(['vgreedyv', 'vgreedy2v', 'mindr'], 'drvp',
# 				rdir='res30_fixi_30')
######## Fig.7 ######## 10 1, 0 8, 22, 16
# compare_traj(['drvp', 'drvg', 'drx'], 'vgreedy2v')
# plot_icurr('vgreedy2v')

# compare_vicontour(['drvp', 'drvg', 'drx'], 'vgreedy2v',
# 				rdir='res30_fixi_30')