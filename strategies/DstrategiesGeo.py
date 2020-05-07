import numpy as np
from geometries import dist, DominantRegion
from math import pi, atan2

from geometries import dist, norm
from strategies.BaseStrategies import BaseStrategy

class DstrategyPPC(BaseStrategy):
	"""docstring for DstrategyPP
	   pure pursuit
	"""
	def __init__(self, game):
		super(DstrategyPPC, self).__init__(game)
	
	def __str__(self):
		return 'ppclose'

	def __call__(self, ss):
		xis, xd, _, _, actives = self.unwrap_state(ss)
		dxs = [xi - xd for xi in xis]
		dists = [dx[0]**2 + dx[1]**2 for dx in dxs]
		min_i = 0
		min_dist = 1e4
		# print(actives) 
		for i, dist in enumerate(dists):
			if bool(actives[i]) and dist < min_dist:
				min_i = i
				min_dist = dist
		# print(min_i)
		return atan2(dxs[min_i][1], dxs[min_i][0])

class DstrategyPPF(BaseStrategy):
	"""docstring for DstrategyPP
	   pure pursuit
	"""
	def __init__(self, game):
		super(DstrategyPPF, self).__init__(game)
		self.reset()
	
	def __str__(self):
		return 'ppfar'

	def __call__(self, ss):

		xis, xd, _, _, actives = self.unwrap_state(ss)
		dxs = [xi - xd for xi in xis]

		if self.last_i is None or self.game.iscap(self.game.intruders[self.last_i]):
			dists = [dx[0]**2 + dx[1]**2 for dx in dxs]
			max_i = 0
			max_dist = 0
			for i, dist in enumerate(dists):
				if bool(actives[i]) and dist > max_dist:
					max_i = i
					max_dist = dist
			i = max_i
			self.last_i = max_i
		else:
			i = self.last_i

		return atan2(dxs[i][1], dxs[i][0])

	def reset(self):
		self.last_i = None


class DstrategyVGreedy(BaseStrategy):
	"""docstring for Vstrategy:
	the defender pick the heading that yeilds the highest value
	for the next step
	"""
	def __init__(self, game, mode='v'):
		super(DstrategyVGreedy, self).__init__(game)
		self.mode = mode
		self.get_dxis = self.get_dxis_fn()

	def __str__(self):
		return 'vgreedy' + self.mode[0]

	def get_dxis_fn(self):
		def from_state(xis, xd, vis, actives):
			return [vi*self.game.dt for vi in vis]
		def from_guess(xis, xd, vis, actives):
			xws = self.get_dr(xis, actives, xd, np.array([0, 0]))
			dxis = [None]*len(xis)
			for i, (xw, xi, active) in enumerate(zip(xws, xis, actives)):
				if active:
					dx = xw - xi
					dxis[i]	= dx/norm(dx)*self.game.vi*self.game.dt
			return dxis
		return from_state if self.mode[-1] == 'v' else from_guess

	def sim_step(self, phis, ss):
		xis, xd, vis, _, actives = self.unwrap_state(ss)
		dxis = self.get_dxis(xis, xd, vis, actives)
		xd_s = [xd + self.game.vd*self.game.dt*np.array([np.cos(phi), np.sin(phi)]) for phi in phis]
		actives_s = [np.array([active for active in actives]) for _ in phis]

		xis_ = [None]*len(xis)
		for i, (xi, active, dxi) in enumerate(zip(xis, actives, dxis)):
			if active:
				xi_ = xi + dxi
				xis_[i] = xi_
				for (xd_, actives_) in zip(xd_s, actives_s):
					if dist(xi_, xd_) < self.r:
						actives_[i] = 0
			else:
				xis_[i] = np.array([x for x in xi])

		ss_s = [np.concatenate(xis_+[xd_]+[np.array([0, 0])]*(self.game.ni+1)+[actives_]) for xd_, actives_ in zip(xd_s, actives_s)]
		return np.vstack(ss_s), np.array([sum(actives) - sum(actives_) for actives_ in actives_s])

	def next_strategy(self, ss):
		phis = np.linspace(-pi, pi, 20)
		ss_s, ncaps = self.sim_step(phis, ss)

		max_ncap = max(ncaps)
		if max_ncap > 0: # if some intruder is captured by a certain phi, pick from these phis
			phi_inds = np.where(ncaps == max_ncap)[0]
			vs = [self.game.value(ss_) for ss_ in ss_s[phi_inds]]
			phi_ind = phi_inds[vs.index(max(vs))]
		else: # if no intruder is captured, pick a phi that gives the highest value
			vs = [self.game.value(ss_) for ss_ in ss_s]
			phi_ind = vs.index(max(vs))

		phi = phis[phi_ind]
		
		return self.wrap_action([phi])[0]

	def ind_strategy(self, ss, i):
		xis, xd, _, _, actives = self.unwrap_state(ss)
		xi = xis[i]

		dr = DominantRegion(self.r, self.a, xi, [xd], offset=0)
		xw = self.target.deepest_point_in_dr(dr)
		dx = xw - xd
		phi = atan2(dx[1], dx[0])

		return self.wrap_action([phi])[0]

	def __call__(self, ss):
		if self.game.ni-self.game.ncap == 1:
			_, _, _, _, actives = self.unwrap_state(ss)
			ind = np.where(actives==1)[0]
			return self.ind_strategy(ss, np.squeeze(ind))
		return self.next_strategy(ss)

class DstrategyVGreedy2(DstrategyVGreedy):
	"""docstring for DstrategyVGreedy2"""
	def __init__(self, game, mode='v'):
		super(DstrategyVGreedy2, self).__init__(game, mode)
	
	def __str__(self):
		return 'vgreedy2'+self.mode

	def next_strategy(self, ss):
		phis = np.linspace(-pi, pi, 20)
		ss_s, ncaps = self.sim_step(phis, ss)
		max_ncap = max(ncaps)
		if max_ncap > 0: # if some intruder is captured by a certain phi, pick from these phis
			phi_inds = np.where(ncaps == max_ncap)[0]
			vs = [self.game.value2(ss_) for ss_ in ss_s[phi_inds]]
			phi_ind = phi_inds[vs.index(max(vs))]
		else: # if no intruder is captured, pick a phi that gives the highest value
			vs = [self.game.value2(ss_) for ss_ in ss_s]
			phi_ind = vs.index(max(vs))
		phi = phis[phi_ind]
		
		return self.wrap_action([phi])[0]


class DstrategyMinDR(BaseStrategy):
	"""docstring for MINDRstrategy"""
	def __init__(self, game):
		super(DstrategyMinDR, self).__init__(game)
		self.reset()

	def __str__(self):
		return 'mindr'

	def new_phi(self, ss):
		xis, xd, _, _, actives = self.unwrap_state(ss)
		xws = self.get_dr(xis, actives, xd, np.array([0, 0]))

		lmin = 100
		for i, (active, xw) in enumerate(zip(actives, xws)):
			if active:
				d2 = self.game.target.level(xw)
				if d2 < lmin:
					lmin = d2
					imin = i
		# print('chasing down intruder', imin)
		dx = xws[imin] - xd
		return atan2(dx[1], dx[0]), imin

	def ind_strategy(self, ss, i):
		xis, xd, _, vd, actives = self.unwrap_state(ss)
		xi = xis[i]

		dr = DominantRegion(self.r, self.a, xi, [xd], offset=0)
		xw = self.target.deepest_point_in_dr(dr)
		dx = xw - xd
		phi = atan2(dx[1], dx[0])

		return self.wrap_action([phi])[0]

	def __call__(self, ss):
		if self.last_i is None or self.game.iscap(self.game.intruders[self.last_i]):
			phi, i = self.new_phi(ss)
		else:
			i = self.last_i
			phi = self.ind_strategy(ss, i)
		self.last_i = i

		return self.wrap_action([phi])[0]

	def reset(self):
		self.last_i = None