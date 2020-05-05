from Games import SDMIGame
from geometries import CircleTarget
from Algorithms import REINFORCE
from Sampler import Sampler

if __name__ == '__main__':

	game = SDMIGame(CircleTarget(1.25), res_dir='res00')
	sampler = Sampler(game)

	reinforce = REINFORCE(sampler, read_dir='regnn')
	reinforce.update(batch_size=32, nsteps=10000)