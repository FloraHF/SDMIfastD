import os
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class RRT(object):
	"""docstring for RRT"""
	class Node(object):
		"""docstring for Node"""
		def __init__(self, xd, xis=[[None, None] for i in range(3)],
					 actives=[True for i in range(3)]):
			self.x = xd[0]
			self.y = xd[1]
			self.xis = [x[0] for x in xis]
			self.yis = [x[1] for x in xis]
			self.actives = [act for act in actives]
			self.path_x = []
			self.path_y = []
			self.path_xi = [[] for _ in range(len(xis))]
			self.path_yi = [[] for _ in range(len(xis))]
			self.parent = None
			self.cost = 0.0

		def get_dist(self):
			if self.parent is not None:
				return math.hypot(self.x - self.parent.x, self.y - self.parent.y)
			return 0

	def __init__(self, game, istrategy,
				 path_resolution=.1,
				 expand_dis=1.5,
				 max_iter=800,
				 min_iter=500,
				 connect_circle_dist=8.0):

		# game parameters
		self.game = game
		self.istrategy = istrategy
		self.pdir = self.game.rdir + 'RRT/'
		if not os.path.exists(self.pdir):
			os.mkdir(self.pdir)

		# RRT parameters
		xis, xd, _, _, actives = game.unwrap_state(game.get_state())
		self.start = self.Node(xd, xis=xis, actives=actives)
		self.start.cost = self.value(self.start)
		self.obstacle_list = []
		self.path_resolution = path_resolution
		self.min_rand = 0.
		self.max_rand = self.game.world_size
		self.expand_dis = expand_dis
		self.connect_circle_dist = connect_circle_dist
		# self.path_resolution = path_resolution
		self.max_iter = max_iter
		self.min_iter = min_iter
		self.node_list = []

	def planning(self, search_until_max_iter=False, draw=False):
		self.node_list = [self.start]
		for i in range(self.max_iter):
			rnd = self.get_random_node()
			nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
			new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)

			if self.check_collision(new_node, self.obstacle_list):
				near_inds = self.find_near_nodes(new_node)
				new_node = self.choose_parent(new_node, near_inds)
				if new_node:
					# print(i, ': get new node')
					self.node_list.append(new_node)
					self.rewire(new_node, near_inds)
					# print(new_node.active)

			if i % 10 == 0:
				print("Iter:", i, ", number of nodes:", len(self.node_list))
				if draw:
					self.draw_graph(rnd)

			if (not search_until_max_iter) and new_node and i > self.min_iter:  # check reaching the goal
				last_index = self.search_best_allcap_node()
				if last_index:
					return self.generate_final_course(last_index)

		print("reached max iteration")

		last_index = self.search_best_allcap_node()
		if last_index:
			return self.generate_final_course(last_index)

		return None, None

	def wrap_state(self, node):
		xis = [np.array([x, y]) for x, y in zip(node.xis, node.yis)]
		xd = [np.array([node.x, node.y])]
		ss = np.concatenate(xis+xd+[np.array([0, 0])]*(self.game.ni+1)+[node.actives])
		return ss
		
	def value(self, node):
		return -self.game.value2(self.wrap_state(node))

	def get_random_node(self):
		rnd = self.Node(np.array([np.random.uniform(self.min_rand, self.max_rand), 
									np.random.uniform(self.min_rand, self.max_rand)]))
		return rnd

	def get_nearest_node_index(self, node_list, rnd_node):
		dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
				 ** 2 for node in node_list]
		minind = dlist.index(min(dlist))
		return minind		

	def calc_distance_and_angle(self, from_node, to_node):
		dx = to_node.x - from_node.x
		dy = to_node.y - from_node.y
		d = math.hypot(dx, dy)
		theta = math.atan2(dy, dx)
		return d, theta
	
	def check_collision(self, node, obstacleList):
		return True

	def steer(self, from_node, to_node, extend_length=float("inf")):
		xd = np.array([from_node.x, from_node.y])
		xis = [np.array([x, y]) for x, y in zip(from_node.xis, from_node.yis)]
		actives = [act for act in from_node.actives]
		new_node = self.Node(xd, xis, actives)

		d, phi = self.calc_distance_and_angle(new_node, to_node)
		psis = self.istrategy(self.wrap_state(new_node))

		new_node.path_x = [new_node.x]
		new_node.path_y = [new_node.y]
		new_node.path_xi = [[x] for x in new_node.xis]
		new_node.path_yi = [[y] for y in new_node.yis]

		if extend_length > d:
			extend_length = d
		if extend_length > self.expand_dis:
			extend_length = self.expand_dis

		dd = 0
		while dd < extend_length:
			dd += self.path_resolution
			xd[0] += self.path_resolution*math.cos(phi)
			xd[1] += self.path_resolution*math.sin(phi)
			cap = False
			for j, (xi, psi) in enumerate(zip(xis, psis)):
				if actives[j]:
					xi[0] += self.path_resolution*math.cos(psi)*self.game.vi/self.game.vd
					xi[1] += self.path_resolution*math.sin(psi)*self.game.vi/self.game.vd
					if (xi[0] - xd[0])**2 + (xi[1] - xd[1])**2 < self.game.r**2:
						actives[j] = False
						cap = True
			if cap:
				break

		new_node.x = xd[0]
		new_node.y = xd[1]
		new_node.path_x.append(new_node.x)
		new_node.path_y.append(new_node.y)
		for i, xi in enumerate(xis):
			new_node.xis[i] = xi[0]
			new_node.yis[i] = xi[1]
			new_node.actives[i] = actives[i]
			new_node.path_xi[i].append(new_node.xis[i])
			new_node.path_yi[i].append(new_node.yis[i])

		new_node.parent = from_node

		return new_node

	def find_near_nodes(self, new_node):
		nnode = len(self.node_list) + 1
		r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
		# if expand_dist exists, search vertices in a range no more than expand_dist
		# if hasattr(self, 'expand_dis'): 
		# 	r = min(r, self.expand_dis)
		dist_list = [(node.x - new_node.x) ** 2 +
					 (node.y - new_node.y) ** 2 for node in self.node_list]
		# print(dist_list, r**2)
		near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
		return near_inds

	def choose_parent(self, new_node, near_inds):
		if not near_inds:
			return None

		# search nearest cost in near_inds
		costs = []
		for i in near_inds:
			near_node = self.node_list[i]
			t_node = self.steer(near_node, new_node)
			if t_node and self.check_collision(t_node, self.obstacle_list):
				costs.append(self.value(t_node))
			else:
				costs.append(float("inf"))  # the cost of collision node
		min_cost = min(costs)

		if min_cost == float("inf"):
			print("There is no good path.(min_cost is inf)")
			return None

		min_ind = near_inds[costs.index(min_cost)]
		new_node = self.steer(self.node_list[min_ind], new_node)
		# new_node.parent = self.node_list[min_ind]
		new_node.cost = min_cost

		return new_node

	def rewire(self, new_node, near_inds):
		for i in near_inds:
			near_node = self.node_list[i]
			edge_node = self.steer(new_node, near_node)
			if not edge_node:
				continue
			edge_node.cost = self.value(edge_node)

			no_collision = self.check_collision(edge_node, self.obstacle_list)
			improved_cost = near_node.cost > edge_node.cost

			if no_collision and improved_cost:
				self.node_list[i] = edge_node
				self.propagate_cost_to_leaves(new_node)

	def propagate_cost_to_leaves(self, parent_node):
		for node in self.node_list:
			if node.parent == parent_node:
				t_node = self.steer(parent_node, node)
				node.cost = self.value(t_node)
				self.propagate_cost_to_leaves(node)

	def search_best_allcap_node(self):
		allcap_inds = []
		for i, n in enumerate(self.node_list):
			if all([not act for act in n.actives]):
				allcap_inds.append(i)

		if allcap_inds:
			costs = [self.node_list[i].cost for i in allcap_inds]
			cost_inds = costs.index(min(costs))
			return np.random.choice([allcap_inds[cost_inds]])
		else:
			# costs = [node.cost for node in self.node_list]
			# return costs.index(min(costs))
			return None

	def generate_final_course(self, goal_ind):
		path = []
		path_i = [[] for _ in range(self.game.ni)]
		node = self.node_list[goal_ind]


		sfile = self.pdir+'state.csv'
		if os.path.exists(sfile):
			os.remove(sfile)
		with open(sfile, 'a') as f:
			f.write('time,'+'state,'*(5*self.game.ni+4)+'\n')

		while node.parent is not None:
			with open(self.pdir+'state.csv', 'a') as f:
				f.write('%.2f,'%(node.get_dist()/self.game.vd) +','.join(map(str, self.wrap_state(node)))+'\n')	
			path.append([node.x, node.y])
			for i in range(self.game.ni):
				path_i[i].append([node.xis[i], node.yis[i]])
			node = node.parent

		with open(self.pdir+'state.csv', 'a') as f:
			f.write('%.2f,'%(node.get_dist()/self.game.vd) +','.join(map(str, self.wrap_state(node)))+'\n')	

		vfile = self.pdir+'value.csv'
		if os.path.exists(vfile):
			os.remove(vfile)
		with open(vfile, 'a') as f:
			f.write('value,%.2f\n'%(-self.value(node)))		

		path.append([node.x, node.y])
		for i in range(self.game.ni):
			path_i[i].append([node.xis[i], node.yis[i]])		

		return path, path_i

	def draw_graph(self, rnd=None):
		plt.clf()
		# # for stopping simulation with the esc key.
		# plt.gcf().canvas.mpl_connect('key_release_event',
		# 							 lambda event: [exit(0) if event.key == 'escape' else None])

		tht = np.linspace(0, 2*math.pi, 50)
		plt.plot(self.game.target.x0 + self.game.target.R*np.cos(tht), self.game.target.y0 + self.game.target.R*np.sin(tht), 'k')
		if rnd is not None:
			plt.plot(rnd.x, rnd.y, "^k")
		for node in self.node_list:
			if node.parent:
				plt.plot(node.path_x, node.path_y, "--g", alpha=0.2)
				# print(node.path_x, node.path_y)
				for i in range(self.game.ni):
					plt.plot(node.path_xi[i], node.path_yi[i], "--r", alpha=0.2)

		for (ox, oy, size) in self.obstacle_list:
			self.plot_circle(ox, oy, size)

		plt.plot(self.start.x, self.start.y, "og")
		for i in range(self.game.ni):
			plt.plot(self.start.xis[i], self.start.yis[i], "or")

		# plt.plot(self.end.x, self.end.y, "xr")
		plt.axis("equal")
		plt.axis([0., 5., 0., 5.])
		plt.grid(True)
		plt.pause(.01)		