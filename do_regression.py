from Games import SDMIGame
from geometries import CircleTarget
from strategies.DstrategiesGeo import DstrategyMinDR
from Algorithms import NNregression
from Sampler import Sampler

if __name__ == '__main__':
	game = SDMIGame(CircleTarget(1.25), res_dir='res00')
	strategy = DstrategyMinDR(game)
	sampler = Sampler(game)

	nnreg = NNregression(sampler, strategy)
	nnreg.update(batch_size=64, nsteps=10000)
