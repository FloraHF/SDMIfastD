import os
import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from geometries import DominantRegion, dist
from strategies.Istrategies import IstrategyDRX

ic = ['r', 'm', 'salmon']


def reward_terminal(game, _s, s_):
	'''step reward that matches the game with only terminal value'''
	return 0

def reward_RL(game, _s, s_):
	'''a reward that is intuitive for RL tasks:
		r =  1 for each intruder captured
		r = -1 for each intruder enters
		r =  0 otherwise
	'''
	_actives = _s[-game.ni:]
	actives_ = s_[-game.ni:]
	ncap = sum(_actives) - sum(actives_)

	xis_ = s_[:-game.ni].reshape(-1, 2)[:-1]
	ents_ = [game.target.level(xi) < 0 for xi in xis_]
	nent = sum(ents_)

	return ncap - nent


class Player(object):

	def __init__(self, env, i, x, color='b'):
		self.env = env
		self.id = i
		self.color = color

		self.reset(x=x)

	def step(self, action):
		if self.active:
			self.v = self.v_max*np.array([np.cos(action), np.sin(action)])
			self.x = self.x + self.env.dt*self.v
		self.path_x.append(self.x[0])
		self.path_y.append(self.x[1])

	@property
	def x(self):
		return self._x
	
	@x.setter
	def x(self, xvalue):
		self._x = np.array([coord for coord in xvalue])

	@property
	def v(self):
		return self._v
	
	@v.setter
	def v(self, vvalue):
		self._v = np.array([coord for coord in vvalue])

	def reset(self, x=np.array([0, 0]), v=np.array([0, 0]), active=1):
		self.active = active
		self.x = x
		self.v = v
		self.path_x = [x[0]]
		self.path_y = [x[1]]
		self.update_status()


class Defender(Player):
	"""docstring for Defender"""
	def __init__(self, env, i, x, color='g'):
		super(Defender, self).__init__(env, i, x, color=color)
		self.v_max = self.env.vd
		self.role = 'D'+str(i)

	def update_status(self): # only used in reset function
		self.icurr = None
		
class Intruder(Player):
	"""docstring for Intruder"""
	def __init__(self, env, i, x, color='r'):
		super(Intruder, self).__init__(env, i, x, color=color)
		self.v_max = self.env.vi
		self.role = 'I'+str(i)

	def update_status(self):
		if self.env.isent(self):
			self.ent = True
			self.active = False
		else: self.ent = False
		if self.env.iscap(self):
			self.cap = True
			self.active = False
		else: self.cap = False

class SDMIGame(object):
	"""docstring for SDMIGame"""
	def __init__(self, target, ni=3, world_size=5.,
						cap_range=.3, vd=4., vi=2., 
						dt=.1,
						xd=np.array([3., 1]), 
						xis=[np.array([0., 2.]), np.array([0., 4.]), np.array([1., 5.])],
						res_dir='res00'):
		self.target = target
		self.world_size = world_size
		self.r = cap_range
		self.ni = ni
		if ni > 1:
			self.orders = [order for order in itertools.permutations([i for i in range(self.ni)])]
		else:
			self.orders = [[0]]
		self.dt = dt

		# player settings
		self.vd = vd
		self.vi = vi
		self.defender = Defender(self, 0, xd)
		self.intruders = [Intruder(self, i, xis[i], color=ic[i]) for i in range(ni)]
		# reset ncap and nent
		self.nent = sum([int(I.ent) for I in self.intruders])
		self.ncap = self.ni - self.nent - sum([int(I.active) for I in self.intruders]) 	# those that are originally inactivate
																						# are treated as captured
		# data recorder settings
		self.rdir = os.path.dirname(__file__) + '/results/' + res_dir +'/'
		self.pfile = 'param.csv'
		self.tfiles = ['traj_I'+str(i)+'.csv' for i in range(ni)] + ['traj_D.csv']
		self.sfile = 'state.csv'

		if not os.path.exists(self.rdir):
			os.mkdir(self.rdir)
		self.record_parm()
	
		self.default_istrategy = IstrategyDRX(self)

	def get_state(self):
		xs = np.concatenate([I.x for I in self.intruders]+[self.defender.x])
		vs = np.concatenate([I.v for I in self.intruders]+[self.defender.v])
		actives = np.array([int(I.active) for I in self.intruders])
		return np.concatenate((xs, vs, actives))

	def unwrap_state(self, ss):
		xs = ss[:2*(self.ni+1)].reshape(-1, 2)
		xis = np.array([[x for x in xy] for xy in xs[:-1]])
		xd = np.array([x for x in xs[-1]])

		vs = ss[2*(self.ni+1):4*(self.ni+1)].reshape(-1, 2)
		vis = np.array([[x for x in xy] for xy in vs[:-1]])
		vd = np.array([x for x in vs[-1]])

		actives = np.array([act for act in ss[-self.ni:]])

		return xis, xd, vis, vd, actives

	def iscap(self, I):
		return (self.defender.x[0] - I.x[0])**2 + (self.defender.x[1] - I.x[1])**2 < self.r**2

	def isent(self, I): # 1.1*self.vd*self.dt handles simulation error
		return self.target.level(I.x) < self.target.minlevel + 1.1*self.vd*self.dt

	def deactivate_ifcap_or_ent(self, i):
		if self.intruders[i].active:
			if self.iscap(self.intruders[i]):
				self.intruders[i].active = False
				self.intruders[i].cap = True
				self.intruders[i].v = np.array([0, 0])
				self.ncap += 1
			if self.isent(self.intruders[i]):
				self.intruders[i].active = False
				self.intruders[i].ent = True
				self.intruders[i].v = np.array([0, 0])
				self.nent += 1

	def record_parm(self):
		if os.path.exists(self.rdir + self.pfile):
			os.remove(self.rdir + self.pfile)
		with open(self.rdir + self.pfile, 'a') as f:
			f.write('r,%.2f\n'%self.r)
			f.write('vd,%.2f\n'%self.vd)
			f.write('vi,%.2f\n'%self.vi)
			f.write('ni,%.2f\n'%self.ni)
			f.write('target,'+str(self.target) + '\n')

	def record_traj(self, t, tfiles, sfile):
		if t == 0:
			for i in range(self.ni):
				with open(tfiles[i], 'a') as f:
					f.write('time,state_0,state_1,state_2,state_3,active\n')
			with open(tfiles[-1], 'a') as f:
				f.write('time,state_0,state_1,state_2,state_3,active\n')
			with open(sfile, 'a') as f:
				f.write('time,'+'state,'*(5*self.ni+4)+'\n')

		for i in range(self.ni):
			with open(tfiles[i], 'a') as f:
				f.write('%.3f,%.3f,%.3f,%.3f,%.3f,%d\n'%(t, 
						self.intruders[i].x[0], self.intruders[i].x[1], 
						self.intruders[i].v[0], self.intruders[i].v[1], 
						int(self.intruders[i].active)))
		with open(tfiles[-1], 'a') as f:
			f.write('%.3f,%.3f,%.3f,%.3f,%.3f,%d\n'%(t, 
					self.defender.x[0], self.defender.x[1], 
					self.defender.v[0], self.defender.v[1], 
					int(self.defender.active)))	
		with open(sfile, 'a') as f:
			f.write('%.3f,'%t +','.join(map(str, self.get_state()))+'\n')	

	def step(self, acts, reward=reward_RL, use_default_istrategy=True):
		_s = self.get_state()
		if use_default_istrategy:
			acts = self.default_istrategy(_s) + [acts]

		self.defender.step(acts[-1])
		for i in range(self.ni):
			self.intruders[i].step(acts[i])
			self.deactivate_ifcap_or_ent(i)

		s_ = self.get_state()
		r = reward(self, _s, s_)
		done = (self.ncap + self.nent >= self.ni)
		info = [self.ncap, self.nent]

		return s_, r, done, info

	def advance(self, T, dstrategy, istrategy=None, record=True, draw=True):
		sdir = self.rdir + str(dstrategy)+'_'+str(istrategy)+'/'
		tfiles = [sdir + tfile for tfile in self.tfiles]
		sfile = sdir + self.sfile
		if not os.path.exists(sdir):
			os.mkdir(sdir)
		for file in tfiles+[sfile]:
			if os.path.exists(file):
				os.remove(file)
		t = 0
		if record:
			self.record_traj(t, tfiles, sfile)
		if istrategy is None:
			istrategy = self.default_istrategy
		for i in range(self.ni):
			self.deactivate_ifcap_or_ent(i)

		while t<T:
			ss = self.get_state()
			actives = np.array([act for act in ss[-self.ni:]])
			if sum(actives) == 0:
				ss_ = ss
				break
			phi = dstrategy(ss)
			psis = istrategy(ss)
			acts = psis + [phi]
			ss_, _, done, _ = self.step(acts, use_default_istrategy=False)
			t += self.dt
			if record:
				self.record_traj(t, tfiles, sfile)
			if draw:
				self.draw_path(dstrategy, istrategy)
			if done:
				plt.show()
				break

		vfile = sdir+'value.csv'
		if os.path.exists(vfile):
			os.remove(vfile)
		with open(vfile, 'a') as f:
			f.write('value,%.2f\n'%self.value(ss_))

	def value_order(self, ss, order):
		xis, xd, _, _, actives = self.unwrap_state(ss)
		t = 0
		xws = []
		for i in order:
			if actives[i]:
				dr = DominantRegion(self.r, self.vd/self.vi, xis[i], [xd], offset=self.vi*t)
				xw = self.target.deepest_point_in_dr(dr)
				dt = dist(xw, xis[i])/self.vi
				t = t + dt
				xd = xd + dt*self.vd*(xw - xd)/dist(xw, xd)
				xws.append(xw)
			else:
				xws.append(np.array([x for x in xis[i]]))
		return min([self.target.level(xw) for xw in xws])	

	def value(self, ss):
		return max([self.value_order(ss, order) for order in self.orders])

	def value2_order(self, ss, order):
		def recurse(xis, actives, xd):
			if sum(actives) == 0:
				return xis, actives, xd
			for k, i in enumerate(order):
				if actives[i]:
					dr = DominantRegion(self.r, self.vd/self.vi, xis[i], [xd], offset=0)
					xw = self.target.deepest_point_in_dr(dr)
					dt = dist(xw, xis[i])/self.vi
					xd = xd + dt*self.vd*(xw - xd)/dist(xw, xd)
					xis[i] = np.array([x for x in xw])
					actives[i] = 0
					break
			for i in order[k+1:]:
				if actives[i]:
					dr = DominantRegion(self.r, self.vd/self.vi, xis[i], [xd], offset=self.vi*dt)
					xw = self.target.deepest_point_in_dr(dr)
					e = (xw - xis[i])/dist(xw, xis[i])
					xi = xis[i] + e*self.vi*dt
					xis[i] = np.array([x for x in xi])
			return recurse(xis, actives, xd)

		# print(ss)
		xis, xd, _, _, actives = self.unwrap_state(ss)
		xis, _, _ = recurse(xis, actives, xd)

		return min([self.target.level(xw) for xw in xis])	

	def value2(self, ss):
		return max([self.value2_order(ss, order) for order in self.orders])

	def reset(self, xd=None, xis=None, actives=None):
		outer_lim = 1.5
		def random_istate(self, xis, actives):
			if xis is None:
				xis = [np.random.uniform(0, self.world_size, size=2) for _ in range(self.ni)]
			for i in range(self.ni):
				while self.target.level(xis[i]) < 0:
					xis[i] = np.random.uniform(0, self.world_size, size=2)			
			if actives is None:
				actives = [np.random.randint(2) for _ in range(self.ni)]
			for I, xi, active in zip(self.intruders, xis, actives):
				I.reset(x=xi, active=active)
			nact = sum([int(I.active) for I in self.intruders])		
			return nact	

		if xd is None:
			xd = np.random.uniform(0, self.world_size, size=2)
			while not 0 < self.target.level(xd) < outer_lim:
				xd = np.random.uniform(0, self.world_size, size=2)
		self.defender.reset(x=xd)

		nact = random_istate(self, xis, actives)
		self.nent = sum([int(I.ent) for I in self.intruders])
		self.ncap = self.ni - self.nent - nact	# those that are originally inactivate 
												# are treated as captured
	def draw_path(self, dstr, istr):
		plt.clf()

		tht = np.linspace(0, 2*np.pi, 50)		
		plt.plot(self.target.x0 + self.target.R*np.cos(tht), self.target.y0 + self.target.R*np.sin(tht), 'k')
		for p in self.intruders + [self.defender]:
			plt.plot(p.path_x, p.path_y, color=p.color, linestyle='--')
			# plt.plot(p.path_x[-p.ntail:], p.path_y[-p.ntail:], p.color)
			plt.plot(p.path_x[-1], p.path_y[-1], p.color, marker='o')

		circle = Circle((self.defender.path_x[-1], self.defender.path_y[-1]), self.r, color=self.defender.color, alpha=0.5)
		plt.gca().add_patch(circle)

		for i in range(self.ni):
			istr.draw_dr(0, i)
			istr.draw_dr(1, i)

		plt.text(2, 5, '_'.join([str(dstr), str(istr)]))
		plt.axis("equal")
		plt.axis([0., 5., 0., 5.])
		plt.grid(True)
		plt.pause(.01)	

