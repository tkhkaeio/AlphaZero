from Coach import Coach
from game.OthelloGame import OthelloGame as Game
from NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 20, #1000 ##
    'numEps': 100, #100
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40, #
    'cpuct': 1,
    'checkpoint': './log',
    'load_model': False,
    'load_folder_file': ('./temp','best.pth.tar'),
    'load_folder_examples': ('./temp','checkpoint_1.pth.tar'), ##
    'save_log_dir': None, 
    'numItersForTrainExamplesHistory': 20,
    'start_iter': 0, ##
    'prevEloScore': [0,0], ##
    'print': True,
})

if __name__=="__main__":
    g = Game(6)
    nnet = nn(g, args)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
