from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
import matplotlib.pyplot as plt
from progress import Bar
from progress.misc import AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import shutil
from distutils.dir_util import copy_tree


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game,  args)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()
        self.elo_score = self.args.prevEloScore #player1_elo, player2_elo
        self.ELO_SCORE = [] #past elo_score
        self.NUM_STEP = [] #number of step: i * numEsp
        
    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
            pi is the MCTS informed policy vector, v is +1 if
            the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b,p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r!=0:
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]

    def learn(self):
        print("start_iter", self.args.start_iter)
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        if not os.path.exists("./log"):
            os.makedirs("log")
        if not os.path.exists("./temp"):
            os.makedirs("temp")
        for i in range(1+self.args.start_iter, self.args.numIters+1+self.args.start_iter):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                eps_time = AverageMeter()
                if self.args.print:
                    bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    if self.args.print:
                        bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                                    total=bar.elapsed_td, eta=bar.eta_td)
                        bar.next()
                if self.args.print:        
                    bar.finish()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i-1)

            # shuffle examlpes before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            # player1: pmcts, player2: nnet
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                        lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game, self.elo_score)
            pwins, nwins, draws, self.elo_score = arena.playGames(self.args.arenaCompare)

            # plot elo score per step
            #self.ELO_SCORE.append(round(self.elo_score[1], 3))
            #self.NUM_STEP.append(10 * i)
            #plotEloScore(i)
            try:
                copy_tree("log/", "/content/drive/My Drive/log")
            except:
                print("fail to copy log")
                pass
            
            print('NEW/PREV WINS : %d / %d ; DRAWS : %d ; ELO SCORE: %.2f(opp: %.2f)' % (nwins, pwins, draws, self.elo_score[1], self.elo_score[0]))
            if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed
        try:
            shutil.copyfile("temp/%s"%filename, "/content/drive/My Drive/temp")
        except:
            print("Fail to copy example")
            pass
        
    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_examples[0], self.args.load_folder_examples[1])
        examplesFile = modelFile+".examples" 
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read %s."%examplesFile)
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
            
    def plotEloScore(self, iter):
        with open("./log/log_%d.txt"%iter, "w") as f:
            f.write("".join([str(step)+"\t"+str(elo)+"\n" for step, elo in zip(self.NUM_STEP, self.ELO_SCORE)]))
        # save loss plots as png
        plt.figure(figsize=(10,5))
        plt.title("Elo score")
        plt.plot(self.NUM_STEP, self.ELO_SCORE,label="elo")
        plt.xlabel("iterations")
        plt.ylabel("Elo")
        plt.legend()
        plt.savefig('./log/log_%d.png'%iter)
        try:
            copy_tree("log/", "/content/drive/My Drive/log")
        except:
            print("fail to copy log")
            pass
