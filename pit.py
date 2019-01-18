#pit.py
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

    
def battle(player1, player2, num):
    arena = Arena.Arena(player1, player2, g, [0,0] , display=display) #player1, player2
    print(arena.playGames(num, verbose=False))
    
    
# nnet players
"""
n1 = NNet(g, argsNN)
n1.load_checkpoint('/content/drive/My Drive/temp/','checkpoint_68.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


n2 = NNet(g, argsNN)
n2.load_checkpoint('/content/drive/My Drive/temp/','checkpoint_1.pth.tar')
args2 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
"""


for p in range(1, 5):
    print("iter:%d"%p)
    for i in range(70, 1, -1):
        n1 = NNet(g, argsNN)
        try:
            n1.load_checkpoint('/content/drive/My Drive/model/','checkpoint_%d.pth.tar'%i)
        except:
            print("no model:%d"%i)
            continue
        args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
        mcts1 = MCTS(g, n1, args1)
        n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0)) #expoitation

        for j in range(70, 1, -1):
            if(i<=j): continue
            n2 = NNet(g, argsNN)
            try:
                n2.load_checkpoint('/content/drive/My Drive/model/','checkpoint_%d.pth.tar'%j)
            except:
                print("no model:%d"%j)
                continue
            args2 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
            mcts2 = MCTS(g, n2, args2)
            n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

            print("model:%d vs model:%d"%(i, j))
            battle(n1p, n2p, 10)
            try:
                copy_tree("model/", "/content/drive/My Drive/model")
            except:
                print("fail to copy model")
                pass