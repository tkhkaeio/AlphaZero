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
    'checkpoint': '/content/drive/My Drive/temp',
    'load_model': True,
    'load_folder_file': ('/content/drive/My Drive/temp','best.pth.tar'),
    'load_folder_examples': ('/content/drive/My Drive/temp','checkpoint_1.pth.tar'), ##
    'save_log_dir': "/content/drive/My Drive/log", 
    'numItersForTrainExamplesHistory': 20,
    'start_iter': 1, ##
    'prevEloScore': [0,0], ##
    'print': False,
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
