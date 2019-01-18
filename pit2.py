import Arena
import torch
from MCTS import MCTS
from game.OthelloGame import OthelloGame, display
from OthelloPlayers import *
from NNet import NNetWrapper as NNet

import numpy as np
from utils import *
import shutil
from distutils.dir_util import copy_tree

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = OthelloGame(6)

argsNN = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play
mmp = MinMaxOthelloPlayer(g).play

# nnet players
n1 = NNet(g, argsNN)
n1.load_checkpoint('./model/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


n2 = NNet(g, argsNN)
n2.load_checkpoint('./model/','checkpoint_5.pth.tar')
args2 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))



arena = Arena.Arena(n1p, mmp, g, [0,0], display=display) 
print(arena.playGames(6, verbose=True))

#arena = Arena.Arena(n2p, mmp, g, [0,0], display=display)
#print(arena.playGames(6, verbose=True))