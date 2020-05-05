import os
import numpy as np
import matplotlib.pyplot as plt

def till_batch(done, counter, batch_size):
	return not done and counter<batch_size

def till_done(done, counter, batch_size):
	return not done

class Sampler(object):
	"""docstring for RegWorker"""
	def __init__(self, game):
		self.game = game

		self.prefix = os.path.dirname(__file__) + '/sampled_data'
		if not os.path.exists(self.prefix):
			os.mkdir(self.prefix)

	def save_data(self, fname, s, a, v):
		if not os.path.exists(fname):
			with open(fname, 'a') as f:
				f.write(','.join(list(map(lambda x: 'state_%s'%x, range(len(s))))+['action','value'])+'\n')
		with open(fname, 'a') as f:
			f.write(','.join(list(map(str, s)) + [str(a), str(v)])+'\n')

	def sample_verification(self, xis, savedata=True):

		buffer_s, buffer_a, buffer_v = [], [], []

		xds = [np.array([x, 0.]) for x in np.linspace(0, 5, 20)]
		for i, xd in enumerate(xds):
			self.game.reset(xis=xis, xd=xd, actives=[1 for _ in xis])
			s = self.game.get_state()
			a = self.strategy(s)
			v = self.game.value(s)
			if savedata:
				self.save_data(self.prefix+'/s_a_v_forveri.csv', s, a[-1], v)

			buffer_s.append(s)
			buffer_a.append(a[-1])
			buffer_v.append(v)

			if i%5 == 0:
				print('number of samples: %s'%i)

		bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(buffer_v)[:, np.newaxis]

		return bs, ba, br

	def sample(self, dstrategy, gmm=1., fname='', batch_size=-1, draw=False, normalize_action=True):
		
		if batch_size <0 : keep_going = till_done
		else: keep_going = till_batch

		self.game.reset()
		s = self.game.get_state()

		buffer_s, buffer_a, buffer_r = [], [], []
		done, counter = False, 0

		while keep_going(done, counter, batch_size): # since there's no running reward, has to reach end
			a = dstrategy(s)
			# print(done)
			s_, r, done, _ = self.game.step(a, use_default_istrategy=True)
			if normalize_action: 
				a = a/np.pi
			# print(a)
			buffer_s.append(s)
			buffer_a.append(a)
			buffer_r.append(r)
			print(r)

			# self.draw_step(s, s_, a)
			s = s_
			counter += 1

		v_s_ = self.game.value(s_)
		# self.draw_value(v_s_)
		discounted_r = []
		for r in buffer_r[::-1]:
			v_s_ = r + gmm * v_s_
			discounted_r.append(v_s_)
		discounted_r.reverse()

		if fname:
			for s, a, r in zip(buffer_s, buffer_a, discounted_r):
				self.save_data(self.prefix+'/'+fname, s, a, r)
		return np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(discounted_r)

	def draw_step(self, s, s_, a):
		plt.clf()
		tht = np.linspace(0, 2*np.pi, 50)		
		plt.plot(self.game.target.x0 + self.game.target.R*np.cos(tht), self.game.target.y0 + self.game.target.R*np.sin(tht), 'k')
		plt.plot(self.game.target.x0 + (self.game.target.R + 1.)*np.cos(tht), self.game.target.y0 + (self.game.target.R + 1.)*np.sin(tht), 'k--')
		plt.plot(self.game.target.x0 + (self.game.target.R + 2.)*np.cos(tht), self.game.target.y0 + (self.game.target.R + 2.)*np.sin(tht), 'k--')
		plt.plot(self.game.target.x0 + (self.game.target.R + 3.)*np.cos(tht), self.game.target.y0 + (self.game.target.R + 3.)*np.sin(tht), 'k--')
		plt.plot([s[0], s_[0]], [s[1], s_[1]], 'r')
		plt.plot([s_[0]], [s_[1]], 'ro')
		# plt.plot([s[2], s_[2]], [s[3], s_[3]], 'r')
		# plt.plot([s_[2]], [s_[3]], 'ro')
		plt.plot([s[2], s_[2]], [s[3], s_[3]], 'g')
		plt.plot([s_[2]], [s_[3]], 'go')

		plt.text(5., 5., 'heading: %.2f deg'%(a*180/3.14))

		plt.axis("equal")
		plt.axis([-1., 6., -1., 6.])
		plt.grid(True)
		plt.pause(.1)	

	def draw_value(self, r):
		plt.text(0., 5., 'value: %.2f'%r)
		plt.pause(0.1)	