import numpy as np
import matplotlib.pyplot as plt

from geometries import DominantRegion, norm
from math import pi, atan2

class BaseStrategy(object):
	"""docstring for BaseStrategy"""
	def __init__(self, game):
		self.game = game
		self.r = game.r
		self.a = game.vd/game.vi
		self.target = game.target
		self.unwrap_state = self.game.unwrap_state
		self.drs = [[None]*game.ni, [None]*game.ni]
		self.xws = [[None]*game.ni, [None]*game.ni]

		self.ls = ['-', '--']

	def get_dr_ind(self, i, xi, active, xd, offvec):
		tier = 1 if norm(offvec)>0 else 0
		if active:
			dr = DominantRegion(self.r, self.a, xi, [xd+offvec], offset=norm(offvec))
			xw = self.target.deepest_point_in_dr(dr)
			self.xws[tier][i] = xw
			self.drs[tier][i] = dr
		else:
			xw = np.array([x for x in xi])
		return xw

	def get_dr(self, xis, actives, xd, offvec):
		# Inputs xd: doesn't include offvec
		xws = []				
		for i, (xi, active) in enumerate(zip(xis, actives)):
			xws.append(self.get_dr_ind(i, xi, active, xd, offvec))
		return xws

	def draw_dr(self, tier, i, ss=None):
		if ss is None: # to be used during playing, otherwise replay (ss reads from file)
			dr = self.drs[tier][i]
			xw = self.xws[tier][i]
			ss = self.game.get_state()
		xis, xd, _, vd, actives = self.unwrap_state(ss)
		# print(xw)
		# if dr is None or xw is None:
		# 	# print('computing tier', tier, 'dr')
		# 	scale = self.game.dt*3*tier # only for tier 0, 1
		# 	if actives[i]:
		# 		dr = DominantRegion(self.r, self.a, xis[i], [xd + vd*scale], offset=self.game.vd*scale)
		# 		xw = self.target.deepest_point_in_dr(dr)

		if dr is not None:
			X, Y, Z = dr.get_data()
			c = plt.contour(X, Y, Z, [0])
			plt.contour(c, colors=(self.game.intruders[i].color), linestyles=(self.ls[tier]))		
		if xw is not None:
			plt.plot(xw[0], xw[1], 
					 color=self.game.intruders[i].color,
					 marker='x')

	def wrap_action(self, acts): # wrap in [-pi, pi]
		for i in range(len(acts)):
			if acts[i] is not None and acts[i] <= -pi:
				n = (-pi - acts[i])%(2*pi) + 1
				acts[i] = acts[i] + (2*pi)*n
			if acts[i] is not None and acts[i] >= pi:
				n = (acts[i] - pi)%(2*pi) + 1
				acts[i] = acts[i] - (2*pi)*n
		return acts