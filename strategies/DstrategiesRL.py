import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class DetermRLstrategy(object):
	def __init__(self, actor):
		self.actor = actor

	def __call__(self, ss):
		phi = np.squeeze(self.actor.predict(np.array([ss])))*np.pi
		print(phi)
		return phi

class StochRLstrategy(object):
	def __init__(self, actor, noise):
		self.actor = actor
		self.noise = noise
		self.dummy_r = np.zeros((1, 1))

	def __call__(self, ss):
		phi = np.squeeze(self.actor.predict([np.array([ss]), self.dummy_r]))*np.pi
		# print(phi)
		bias = np.random.normal(scale=self.noise)
		return np.clip(phi + bias, -np.pi, np.pi)


class RLvalue(object):
	def __init__(self, model):
		self.model = model

	def __call__(self, ss):
		return np.squeeze(self.model.predict(np.array([ss])))