import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from keras import backend as K
# tf.compat.v1.disable_eager_execution()

from Sampler import Sampler
from strategies.DstrategiesRL import StochRLstrategy


class REINFORCE(object):
	"""docstring for PolicyFromValue"""
	def __init__(self, sampler, save_dir='REINFORCE', read_dir=''):
		
		self.sampler = sampler
		self.optimizer=tf.keras.optimizers.Adam()
		self.save_dir = save_dir
		if not read_dir: self.read_dir = save_dir
		else: self.read_dir = read_dir

		self.noise = .2

		self.actor = self.initiate_model('actor')
		self.strategy = StochRLstrategy(self.actor, self.noise)

	def initiate_model(self, name):
		prefix = os.path.dirname(__file__)
		if not os.path.exists(prefix+'/'+self.save_dir):
			os.mkdir(prefix+'/'+self.save_dir)
		if not os.path.exists(prefix+'/'+self.save_dir+'/'+name):
			os.mkdir(prefix+'/'+self.save_dir+'/'+name)

		if not os.listdir(prefix+'/'+self.read_dir+'/'+name):
			print('----------->> creating new ' + name + ' >>>>>>>>>>>>>>')
			model = self.get_actor()			
		else:
			print('----------->> loading ' + name + ' from save: '+ self.read_dir+'/'+name + ' >>>>>>>>>>>>>>')
			model = tf.keras.models.load_model(self.read_dir+'/'+name)	
		return model

	def get_actor(self):
		x_in = tf.keras.Input(shape=(5,))
		temp = tf.keras.layers.Dense(32, activation='relu')(x_in)
		a_pred = tf.keras.layers.Dense(1, activation='tanh')(temp)

		# r = tf.keras.Input(shape=(1,))
		model = tf.keras.models.Model(inputs=x_in, outputs=a_pred)
		model.summary()
		return model

	@tf.function
	def loss_fn(self, r, y_true, y_pred):
		r = tf.cast(r, tf.float32)
		y_pred = tf.cast(y_pred, tf.float32)
		y_true = tf.cast(y_true, tf.float32)
		var = K.square(self.noise)
		pi = 3.1415926
		denom = K.sqrt(2 * pi * var)
		prob = K.exp(- K.square(y_true - y_pred) / (2 * var))
		# print(prob)
		log_prob = K.log(prob)
		return -K.mean(r*log_prob)
 
	def update(self, batch_size=32, nsteps=10):

		err_wd, err_hist = [], []
		step = 0

		while step < nsteps:
			bs, ba, br = self.sampler.sample(self.strategy)
			while len(bs) < batch_size:
				s, a, adv = self.sampler.sample(self.strategy)
				bs = np.vstack((bs, s))
				ba = np.vstack((ba, a))
				br = np.vstack((br, adv))
			bs = bs[:batch_size] 
			ba = ba[:batch_size]
			br = br[:batch_size]
			# print(a)
			# print(bs[0], ba[0], br[0])
			with tf.GradientTape() as g:
				loss = self.loss_fn(br, ba, self.actor(bs))

			trainable_variables = self.actor.trainable_variables
			gradients = g.gradient(loss, trainable_variables)
			self.optimizer.apply_gradients(zip(gradients, trainable_variables))

			step += 1
			self.log_result(step, err_wd, err_hist, loss)

	def log_result(self, step, err_wd, err_hist, loss):
		if len(err_wd) >= 20:
			err_wd.pop(0)
		err_wd.append(loss)
		err_av = sum(err_wd)/len(err_wd)
		if len(err_hist)>20000:
			err_hist.pop(0)				
		err_hist.append(err_av)

		if step%10 == 0:
			print('training step: %d | loss: %.8f (ave: %.8f)'%(step, loss, err_av))
		if step%10 == 0:
			self.draw_res(err_hist)
			self.actor.save(self.save_dir+'/actor')	


	def draw_res(self, err_hist):
		plt.clf()
		plt.plot(range(len(err_hist)), err_hist, 'b', label='actor')
		plt.legend()
		plt.grid(True)
		plt.savefig('convREINFORCE.jpg')	


class NNregression(object):
	"""docstring for NNregression"""
	def __init__(self, sampler, strategy, save_dir='regnn'):

		self.save_dir = save_dir
		self.sampler = sampler
		self.game = sampler.game
		self.strategy = strategy

		self.actor = self.initiate_model('actor')
		self.critic = self.initiate_model('critic')

	def initiate_model(self, name):
		prefix = os.path.dirname(__file__)
		if not os.path.exists(prefix+'/'+self.save_dir):
			os.mkdir(prefix+'/'+self.save_dir)
		if not os.path.exists(prefix+'/'+self.save_dir+'/'+name):
			os.mkdir(prefix+'/'+self.save_dir+'/'+name)
		if not os.listdir(prefix+'/'+self.save_dir+'/'+name):
			print('----------->> creating new ' + name + ' >>>>>>>>>>>>>>')
			if 'actor' in name:
				model = self.get_actor()			
			if 'critic' in name:
				model = self.get_critic()
		else:
			print('----------->> loading ' + name + ' from save >>>>>>>>>>>>>>')
			model = tf.keras.models.load_model(self.save_dir+'/'+name)	
		return model

	def get_actor(self):
		x_in = tf.keras.Input(shape=(5,))
		temp = tf.keras.layers.Dense(32, activation='relu')(x_in)
		a_pred = tf.keras.layers.Dense(1, activation='tanh')(temp)
		model = tf.keras.models.Model(inputs=x_in, outputs=a_pred)
		model.compile(optimizer='Adam', loss='mean_squared_error')
		model.summary()
		return model

	def get_critic(self):
		x_in = tf.keras.Input(shape=(5,))
		temp = tf.keras.layers.Dense(32, activation='relu')(x_in)
		value = tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(temp)
		model = tf.keras.models.Model(inputs=x_in, outputs=value)
		model.compile(optimizer='Adam', loss='mean_squared_error')
		model.summary()
		return model

	def update(self, data_file='sar.csv', batch_size=32, nsteps=100):

		aerr_wd, aerr_hist = [], []
		cerr_wd, cerr_hist = [], []
		for i in range(nsteps):
			bs, ba, br = self.sampler.sample(self.strategy, fname='', batch_size=1) # sample for 1 step
			while len(bs) < batch_size:
				# print('new data')
				s, a, r = self.sampler.sample(self.strategy, fname='', batch_size=1)
				bs = np.vstack((bs, s))
				ba = np.vstack((ba, a))
				br = np.vstack((br, r))
			bs = bs[:batch_size]
			ba = ba[:batch_size]
			br = br[:batch_size]	

			closs = self.critic.train_on_batch(bs, br)
			aloss = self.actor.train_on_batch(bs, ba)
			# print(self.actor.predict([[bs[0]]]))

			self.log_result(i, aerr_wd, aerr_hist, aloss, 
								  cerr_wd, cerr_hist, closs)

	def log_result(self, step, aerr_wd, aerr_hist, aloss, cerr_wd, cerr_hist, closs):
		if len(aerr_wd) >= 20:
			aerr_wd.pop(0)
		aerr_wd.append(aloss)
		aerr_av = sum(aerr_wd)/len(aerr_wd)
		if len(aerr_hist)>20000:
			aerr_hist.pop(0)				
		aerr_hist.append(aerr_av)

		if len(cerr_wd) >= 20:
			cerr_wd.pop(0)
		cerr_wd.append(closs)
		cerr_av = sum(cerr_wd)/len(cerr_wd)
		if len(cerr_hist)>20000:
			cerr_hist.pop(0)				
		cerr_hist.append(cerr_av)

		if step%10 == 0:
			print('training step: %d | aloss: %.8f (ave: %.8f) | closs: %.8f (ave: %.8f)'%(step, aloss, aerr_av, aloss, cerr_av))
		if step%10 == 0:
			self.draw_res(aerr_hist, cerr_hist)
			self.actor.save(self.save_dir+'/actor')
			self.critic.save(self.save_dir+'/critic')
			
	def draw_res(self, aerr_hist, cerr_hist):
		plt.clf()
		plt.plot(range(len(aerr_hist)), aerr_hist, 'b', label='actor')
		plt.plot(range(len(cerr_hist)), cerr_hist, 'r', label='critic')
		plt.legend()
		plt.grid(True)
		plt.savefig('convReg.jpg')	

class NstepTDprediction(object):
	"""docstring for TDprediction"""
	def __init__(self, sampler, strategy, save_dir='nsteptd'):

		self.sampler = sampler
		self.strategy = strategy # the strategy of which the value function is learned
		
		self.loss = tf.keras.losses.MeanSquaredError()
		self.optimizer = tf.keras.optimizers.Adam()

		self.save_dir = save_dir
		prefix = os.path.dirname(__file__)
		if os.path.exists(prefix+'/'+save_dir):
			self.vfn = tf.keras.models.load_model(save_dir)
		else:
			os.mkdir(prefix+'/'+save_dir)
			self.vfn = build_a_net('vfn')
		
		self.err_hist = []

	def update(self, data_file='', batch_size=32, nsteps=10):

		err_wd = []
		step = 0

		if data_file:
			data = pd.read_csv(os.path.dirname(__file__)+'/sampled_data/'+data_file)

		while step < nsteps:

			if data_file:
				batch = data.sample(batch_size)
				bs = batch.filter(regex='state').to_numpy()
				br = batch.filter(regex='value').to_numpy()	
			else:
				bs, ba, br = self.sampler.sample_episode(self.strategy, fname='')
				while len(bs) < batch_size:
					s, a, r = self.sampler.sample_episode(self.strategy, fname='')
					bs = np.vstack((bs, s))
					br = np.vstack((br, r))
				bs = bs[:batch_size]
				br = br[:batch_size]		

			with tf.GradientTape() as g:
				td_err = self.loss(self.vfn(bs), br)
				trainable_variables = self.vfn.trainable_variables
				gradients = g.gradient(td_err, trainable_variables)
				self.optimizer.apply_gradients(zip(gradients, trainable_variables))

			step += 1
			if len(err_wd) >= 20:
				err_wd.pop(0)
			err_wd.append(td_err)
			err_av = sum(err_wd)/len(err_wd)

			if len(self.err_hist)>2000:
				self.err_hist.pop(0)
			self.err_hist.append(err_av)

			if step%20 == 0:
				print('training step: %d | loss: %.3f | average loss: %.3f'%(step, td_err, err_av))

			if step%100 == 0:	
				self.draw_res()
				self.vfn.save(self.save_dir)

	def draw_res(self):
		plt.clf()
		plt.plot(range(len(self.err_hist)), self.err_hist)
		plt.grid(True)
		plt.savefig('convTD.jpg')


class PolicyForGameValuePPO(object):
	"""docstring for PolicyFromValue"""
	def __init__(self, game, sampler, save_dir='piforgv'):
		
		self.sampler = sampler
		self.optimizer = tf.keras.optimizers.Adam()

		self.save_dir = save_dir

		prefix = os.path.dirname(__file__)
		if os.path.exists(prefix+'/'+save_dir):
			self.pfn = tf.keras.models.load_model(save_dir)
			self.pfn_ = tf.keras.models.load_model(save_dir)
		else:
			os.mkdir(prefix+'/'+save_dir)
			self.pfn = build_stochastic_pnet('pfn', trainable=True)
			self.pfn_ = build_stochastic_pnet('pfn_', trainable=False)
		# self.pfn = build_stochastic_pnet('pfn', trainable=True)
		# self.pfn_ = build_stochastic_pnet('pfn_', trainable=False)
		
		from RLstrategies import StochRLstrategy
		self.strategy = StochRLstrategy(game, self.pfn)

		self.normdist = tfp.distributions.Normal(loc=0., scale=1.)
		self.eps = 0.2

		self.err_hist = []

	@tf.function
	def loss(self, s, a, adv):
		mu, sigma = self.pfn(s)
		mu_, sigma_ = self.pfn_(s)
		prob = self.normdist.prob(tf.cast((a - mu)/(sigma + 1e-5), tf.float32))
		prob_ = self.normdist.prob(tf.cast((a - mu_)/(sigma_ + 1e-5), tf.float32))
		ratio = tf.cast(tf.divide(prob, prob_ + 1e-5), tf.float64)
		surr = tf.multiply(ratio, adv)

		loss = -tf.reduce_mean(tf.minimum(surr, 
								tf.clip_by_value(ratio, 1.-self.eps, 1.+self.eps)*adv))
		return loss

	def update(self, batch_size=32, nsteps=10):

		err_wd = []
		step = 0

		while step < nsteps:
			bs, ba, b_adv = self.sampler.sample_batch(self.strategy, adv=True)
			while len(bs) < batch_size:
				s, a, adv = self.sampler.sample_batch(self.strategy, adv=True)
				bs = np.vstack((bs, s))
				ba = np.vstack((ba, a))
				b_adv = np.vstack((b_adv, adv))
			bs = bs[:batch_size]
			ba = ba[:batch_size]
			b_adv = b_adv[:batch_size]		

			self.pfn_.set_weights(self.pfn.get_weights())

			with tf.GradientTape() as g:
				loss = self.loss(bs, ba, b_adv)
				trainable_variables = self.pfn.trainable_variables
				gradients = g.gradient(loss, trainable_variables)
				self.optimizer.apply_gradients(zip(gradients, trainable_variables))

			step += 1
			if len(err_wd) >= 20:
				err_wd.pop(0)
			err_wd.append(loss)
			err_av = sum(err_wd)/len(err_wd)
			if len(self.err_hist)>2000:
				self.err_hist.pop(0)
			self.err_hist.append(err_av)

			if step%20 == 0:
				print('training step: %d | loss: %.3f | average loss: %.3f'%(step, loss, err_av))

			if step%20 == 0:
				self.draw_res()
				self.pfn.save(self.save_dir)		

	def draw_res(self):
		plt.clf()
		plt.plot(range(len(self.err_hist)), self.err_hist)
		plt.grid(True)
		plt.savefig('convPPOpi.jpg')


class PPO(object):
	"""docstring for PolicyFromValue"""
	def __init__(self, game, sampler, save_dir='PPO', pdir='', vdir=''):
		
		self.sampler = sampler
		self.coptimizer = tf.keras.optimizers.Adam()
		self.aoptimizer = tf.keras.optimizers.Adam()

		self.save_dir = save_dir

		prefix = os.path.dirname(__file__)
		if not os.path.exists(prefix+'/'+save_dir):
			os.mkdir(prefix+'/'+save_dir)

		if os.path.exists(prefix+'/'+save_dir+'/ValueFn'): 	# load former saved results if exits
			self.critic = tf.keras.models.load_model(save_dir+'/ValueFn')
		else:
			os.mkdir(save_dir+'/ValueFn')
			if vdir: 			# load results from n-step td if exists
				self.critic = tf.keras.models.load_model(vdir)
			else: 											# build a new net otherwise
				self.critic = build_a_net('critic')
		# self.critic.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

		if os.path.exists(prefix+'/'+save_dir+'/PolicyFn'):	# load former saved results if exits
			self.actor = tf.keras.models.load_model(save_dir+'/PolicyFn')
			self.actor_ = tf.keras.models.load_model(save_dir+'/PolicyFn')
		else:
			os.mkdir(save_dir+'/PolicyFn')
			if pdir: 			# load results from n-step td if exists
				self.actor = tf.keras.models.load_model(pdir)
				self.actor_ = tf.keras.models.load_model(pdir)
			else: 											# build a new net otherwise
				self.actor = build_stochastic_pnet('actor', trainable=True)
				self.actor_ = build_stochastic_pnet('actor_', trainable=False)

		from RLstrategies import StochRLstrategy, RLvalue
		self.strategy = StochRLstrategy(game, self.actor)
		self.value = RLvalue(self.critic)

		self.normdist = tfp.distributions.Normal(loc=0., scale=1.)
		self.eps = 0.2

		self.closs = tf.keras.losses.MeanSquaredError()

		self.aerr_hist = []
		self.cerr_hist = []

	@tf.function
	def aloss(self, s, a, r): # loss function for actor
		mu, sigma = self.actor(s)
		mu_, sigma_ = self.actor_(s)
		prob = self.normdist.prob(tf.cast((a - mu)/(sigma + 1e-5), tf.float32))
		prob_ = self.normdist.prob(tf.cast((a - mu_)/(sigma_ + 1e-5), tf.float32))
		adv = r - tf.cast(self.critic(s), tf.float64)
		ratio = tf.cast(tf.divide(prob, prob_ + 1e-5), tf.float64)
		surr = tf.multiply(ratio, adv)

		loss = -tf.reduce_mean(tf.minimum(surr, 
								tf.clip_by_value(ratio, 1.-self.eps, 1.+self.eps)*adv))
		return loss

	def update(self, batch_size=32, nsteps=10):

		aerr_wd, cerr_wd = [], []
		step = 0

		while step < nsteps:
			bs, ba, br = self.sampler.sample_episode(self.strategy)
			while len(bs) < batch_size:
				s, a, adv = self.sampler.sample_episode(self.strategy)
				bs = np.vstack((bs, s))
				ba = np.vstack((ba, a))
				br = np.vstack((br, adv))
			bs = bs[:batch_size]
			ba = ba[:batch_size]
			br = br[:batch_size]		

			self.actor_.set_weights(self.actor.get_weights())

			# self.optimizer.minimize(self.aloss, self.actor.trainable_variables)
			# self.optimizer.minimize(self.closs, self.critic.trainable_variables)
			with tf.GradientTape() as g:
				aloss = self.aloss(bs, ba, br)
				trainable_variables = self.actor.trainable_variables
				gradients = g.gradient(aloss, trainable_variables)
				self.aoptimizer.apply_gradients(zip(gradients, trainable_variables))

			with tf.GradientTape() as g:
				closs = self.closs(self.critic(bs), br)
				trainable_variables = self.critic.trainable_variables
				gradients = g.gradient(closs, trainable_variables)
				self.coptimizer.apply_gradients(zip(gradients, trainable_variables))

			step += 1
			if len(cerr_wd) >= 20:
				cerr_wd.pop(0)
			if len(aerr_wd) >= 20:				
				aerr_wd.pop(0)
			aerr_wd.append(aloss)
			cerr_wd.append(closs)
			aerr_av = sum(aerr_wd)/len(aerr_wd)
			cerr_av = sum(cerr_wd)/len(cerr_wd)
			if len(self.aerr_hist)>20000:
				self.aerr_hist.pop(0)
			if len(self.cerr_hist)>10000:
				self.cerr_hist.pop(0)				
			self.aerr_hist.append(aerr_av)
			self.cerr_hist.append(cerr_av)

			if step%20 == 0:
				print('training step: %d | aloss: %.8f (ave: %.8f) | closs:%.8f (ave: %.8f)'%(step, aloss, aerr_av, closs, cerr_av))
				
			if step%100 == 0:
				self.draw_res()
				self.actor.save(self.save_dir+'/PolicyFn')
				self.critic.save(self.save_dir+'/ValueFn')		

	def draw_res(self):
		plt.clf()
		plt.plot(range(len(self.aerr_hist)), self.aerr_hist, 'b', label='actor')
		plt.plot(range(len(self.cerr_hist)), self.cerr_hist, 'g', label='critic')
		plt.legend()
		plt.grid(True)
		plt.savefig('convPPO.jpg')
