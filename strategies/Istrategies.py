import numpy as np
from math import atan2, cos, sin

from strategies.BaseStrategies import BaseStrategy
from geometries import DominantRegion, dist, norm

class IstrategyDT(BaseStrategy):
	"""docstring for IstrategyDT
	   the intruder heads to the center of the target directly"""
	def __init__(self, game):
		super(IstrategyDT, self).__init__(game)

	def __str__(self):
		return 'dt'
	
	def __call__(self, ss):
		xis, xd, _, vd, actives = self.unwrap_state(ss)
		xt = np.array([self.target.x0, self.target.y0])
		
		dxs = [xt - xi for xi in xis]
		psis = [atan2(dx[1], dx[0]) for dx in dxs]
		return self.wrap_action(psis)
		

class IstrategyDRX(BaseStrategy):
	"""docstring for IstrategyDRX
	   intruder's strategy that takes in to consider
	   the defender's location
	"""
	def __init__(self, game):
		super(IstrategyDRX, self).__init__(game)

	def __str__(self):
		return 'drx'

	def get_psis(self, xis, actives, xd):
		xws = self.get_dr(xis, actives, xd, np.array([0, 0]))
		psis = [None]*len(xis)
		for i, (xw, xi, active) in enumerate(zip(xws, xis, actives)):
			if active:
				dx = xw - xi
				psis[i] = atan2(dx[1], dx[0])		
		return psis

	def __call__(self, ss):
		xis, xd, _, _, actives = self.unwrap_state(ss)
		psis = self.get_psis(xis, actives, xd)
		return self.wrap_action(psis)


class IstrategyDRV(BaseStrategy):
	"""docstring for IstrategyDRV
	   intruder's strategy that takes in to consider
	   the defender's location AND velocity
	  """

	def __init__(self, game, mode='pred'):
		super(IstrategyDRV, self).__init__(game)
		self.mode = mode
		self.call = self.get_action()

	def __str__(self):
		return 'drv' + self.mode[0]

	def estimate_icurr(self, xis, actives, xd, vd):
		xws = self.get_dr(xis, actives, xd, np.array([0, 0]))
		ed = np.array([0, 0]) if norm(vd) == 0 else vd/norm(vd)
		err = [None]*len(xis)
		for i, (xw, active) in enumerate(zip(xws, actives)):
			if active:
				dx = xw - xd
				e = dx/norm(dx)
				err[i] = e[0]*ed[0] + e[1]*ed[1]
		emax = -1
		for i, e in enumerate(err):
			if e is not None and e >= emax:
				emax = e
				imax = i
		return imax, xws[imax]

	def get_action(self):
		def greedy(ss):
			xis, xd, _, vd, actives = self.unwrap_state(ss)
			xws = self.get_dr(xis, actives, xd, vd*self.game.dt*10)
			psis = [None]*len(xis)
			for i, (xi, xw, active) in enumerate(zip(xis, xws, actives)):
				if active:
					dx = xw - xi
					psis[i] = atan2(dx[1], dx[0])
			return self.wrap_action(psis)

		def long_term(ss):
			xis, xd, _, vd, actives = self.unwrap_state(ss)
			imax, xw_imax = self.estimate_icurr(xis, actives, xd, vd)

			d = dist(xw_imax, xd)
			offvec = (xw_imax - xd)*(d - self.game.r)/d 
			psis = [None]*len(xis)
			for i, (xi, active) in enumerate(zip(xis, actives)):
				if active:
					if i == imax:
						dx = xw_imax - xi
					else:
						xw = self.get_dr_ind(i, xi, active, xd, offvec)
						dx = xw - xi
					psis[i] = atan2(dx[1], dx[0])
			return self.wrap_action(psis)

		return long_term if self.mode == 'pred' else greedy

	def __call__(self, ss):
		return self.call(ss)
