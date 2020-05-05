import argparse
import numpy as np 
import pandas as pd
import csv
# import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from Games import SDMIGame
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
fs = 14
colors = ['red', 'blue', 'green', 'm', 'k']

def read_data(dr):
	print('reading from', dr)
	data = pd.read_csv(dr+'/state.csv')
	t = data.filter(regex='time').to_numpy()
	ss = data.filter(regex='state').to_numpy()
	# n = int(rdir.split('_')[0][-2])
	n = int((len(ss[0]) - 4)/5)
	xd = pd.read_csv(dr+'/traj_D.csv').filter(regex='state').to_numpy()
	xis = [pd.read_csv(dr+'/traj_I'+str(i)+'.csv').filter(regex='state').to_numpy() for i in range(n)]
	actives = ss[:,-n:]
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
	return n, t, xis, xd, ss, capids, value

def plot_target():
	tht = np.linspace(0, 2*np.pi, 50)		
	plt.plot(game.target.x0 + R*np.cos(tht), 
			 game.target.y0 + R*np.sin(tht), 
			 'k', linewidth=1.5, label='Target')

def compare_traj(stras, base, trange=-1):
	fig = plt.figure()
	plot_target()
	
	for stra, c in zip(stras, colors):
		# print(stra)
		try: n, t, xis, xd, _, capids, value = read_data('results/'+rdir+'/'+'_'.join([stra, base]))
		except: n, t, xis, xd, _, capids, value = read_data('results/'+rdir+'/'+'_'.join([base, stra]))
		if trange == -1:
			trange = len(t)-1
		print(capids)

		# plot trajectories
		plt.plot(xd[:trange,0], xd[:trange,1], color=c, linewidth=1.5, linestyle=(0,()), label=stra+' v='+value)
		for i in range(n):
			plt.plot(xis[i][:trange,0], xis[i][:trange,1], color=c, linewidth=1.5, linestyle=(0, (8,3)))

		# plot markers
		for i, k in enumerate(capids+[trange]):
			if k <= trange:
				plt.plot(xd[k,0], xd[k,1], color=c, marker='o')
				if i != 0:
					circle = Circle(xd[k], r, color=c, alpha=0.2)
					plt.gca().add_patch(circle)
				for i in range(n):
					plt.plot(xis[i][k,0], xis[i][k,1], color=c, marker='>')
	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.axis("equal")
	plt.axis([0., 5., 0., 5.])
	plt.xlabel('x', fontsize=fs)
	plt.ylabel('y', fontsize=fs)
	plt.legend(fontsize=fs, loc='best')
	plt.show()
	# plt.savefig('results/'+rdir+'/traj_'+dstr+'.png')
	# plt.close()

def compare_value(stras, base):
	fig = plt.figure()
	
	for stra, c in zip(stras, colors):
		# print(stra)
		try: _, t, _, _, ss, _, _ = read_data('results/'+rdir+'/'+'_'.join([stra, base]))
		except: _, t, _, _, ss, _, _ = read_data('results/'+rdir+'/'+'_'.join([base, stra]))

		plt.plot(t, [game.value2(s) for s in ss], label=stra, color=c)
		# plt.plot(t, [game.value(s) for s in ss], label=stra, color=c, linestyle='--')

	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.xlabel('t', fontsize=fs)
	plt.ylabel('value', fontsize=fs)
	
	plt.legend(fontsize=fs, ncol=2, loc='best')
	# plt.savefig('results/'+rdir+'/maxvalue_'+str(istr)+'.png')
	# plt.close()
	plt.show()

def compare_order(dstr, istr):
	fig = plt.figure()
	_, t, _, _, ss, _, _ = read_data('results/'+rdir+'/'+'_'.join([dstr, istr]))

	for order, c in zip(game.orders, ['r', 'b', 'k', 'm', 'g', 'c']):
		plt.plot(t, [game.value2_order(s, order) for s in ss], color=c, linestyle='--', label=','.join(map(str,order)))
	plt.plot(t, [game.value2(s) for s in ss], color='b', linewidth=2, label='value')	

	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.xlabel('t', fontsize=fs)
	plt.ylabel('value', fontsize=fs)
	plt.legend(fontsize=fs)
	# plt.savefig('results/'+rdir+'/value_'+'_'.join([dstr, istr])+'.png')
	# plt.close()
	plt.show()

def compare_value12(dstr, istr):
	fig = plt.figure()
	_, t, _, _, ss, _, _ = read_data('results/'+rdir+'/'+'_'.join([dstr, istr]))

	# for order, c in zip(game.orders, ['r', 'b', 'k', 'm', 'g', 'c']):
	# 	plt.plot(t, [game.value_order(s, order) for s in ss], color=c, linestyle='--', label=','.join(map(str,order)))
	plt.plot(t, [game.value(s) for s in ss], 'r', label='vgreedy')	
	plt.plot(t, [game.value2(s) for s in ss], 'g', label='vgreedy2')	

	plt.grid()
	plt.gca().tick_params(axis='both', which='major', labelsize=fs)
	plt.gca().tick_params(axis='both', which='minor', labelsize=fs)
	plt.xlabel('t', fontsize=fs)
	plt.ylabel('value', fontsize=fs)
	plt.legend(fontsize=fs)
	# plt.savefig('results/'+rdir+'/value_'+'_'.join([dstr, istr])+'.png')
	# plt.close()
	plt.show()


# n = 1
# compare_traj(['mindr', 'ppfar', 'ppclose'], 'drx')
# compare_traj(['drvg', 'drx'], 'ppclose')
# compare_value(['drx', 'drvg', 'dt'], 'mindr')

# n >= 2
compare_traj(['vgreedyx', 'vgreedyv', 'vgreedy2', 'mindr'], 'drx')
compare_traj(['mindr', 'ppf', 'ppc'], 'drx')
compare_traj(['drx', 'drvp', 'drvg', 'dt'], 'mindr')
compare_traj(['drx', 'drvp', 'drvg', 'dt'], 'vgreedy2')

compare_value(['vgreedy', 'vgreedy2', 'mindr'], 'drx')
compare_value(['mindr', 'ppf', 'ppc'], 'drx')
compare_value(['drx', 'drvp', 'drvg'], 'mindr')
compare_value(['drx', 'drvp', 'drvg'], 'vgreedy2')

compare_order('vgreedy2', 'drvp')

compare_value12('vgreedy', 'drvp')



