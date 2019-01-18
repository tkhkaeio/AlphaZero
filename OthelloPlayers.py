%%writefile OthelloPlayers.py
import numpy as np

INFINITY = 10000000


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.n), int(i%self.game.n))
        while True:
            a = input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class GreedyOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)] #min
        candidates.sort()
        return candidates[0][1]


class MinMaxOthelloPlayer():
    def __init__(self, game):
        self.game = game
        self.boardSize = self.game.getBoardSize()[0]
        self.gradingStrategy = [
            0,   0,   0,   0,   0,   0,   0,   0,
            0, 100, -10,  10,  10, -10, 100,   0,
            0, -10, -20,  -3,  -3, -20, -10,   0,
            0,  10,  -3,   1,   1,  -3,  10,   0,
            0,  10,  -3,   1,   1,  -3,  10,   0,
            0, -10, -20,  -3,  -3, -20, -10,   0,
            0, 100, -10,  10,  10, -10, 100,   0,
            0,   0,   0,   0,   0,   0,   0,   0,
        ]#It's a board of weight of each position. Corners and most edges are important. 
    
    def eval_fn(self, board, player):
        # if the game is over, give a 100 point bonus to the winning player
        if self.game.getGameEnded(board, player):
            point = self.game.getScore(board, player)
            if point > 0:
                return 100
            elif point < 0:
                return -100
            else:
                return 0
        point = 0
        #find the player of the opponent
        for row in range(self.boardSize):
            for col in range(self.boardSize):
                #calculate the point of current player
                if board[row][col] == player:
                    point += self.gradingStrategy[(row+1)*(self.boardSize+2)+1+col]
                #calculate the point of the opponent
                elif board[row][col] == -player:
                    point -= self.gradingStrategy[(row+1)*(self.boardSize+2)+1+col]
        return point

    def minimax(self, board, player, depth):
        #Find the best move in the game
        #if depth = 0, we calculate the score
        if depth == 0:
            return self.eval_fn(board, player)
        #if game is over, we calculate the score
        if self.game.getGameEnded(board, player):
            return self.game.getScore(board, player)

        best_val = None
        best_move = None
        opp = -player
        # valid moves
        moves = self.game.getValidMoves(board, player)
        #shuffle the moves in case it places the same position in every game
        #shuffle(moves)
        if moves[-1] == 1:
            return self.eval_fn(board, player) #pass
        #try each move in valid moves
        #evaluate max's position and choose the best value
        if player == 1:
            for a in range(self.game.getActionSize()):
                if moves[a]==0:
                    continue
                nextBoard, _ = self.game.getNextState(board, player, a)
                val = self.minimax(nextBoard, opp, depth-1)
                if best_val is None or val > (best_val, best_move)[0]:
                    (best_val, best_move) = (val, a)
        #evaluate min's position and choose the best value
        if player == -1:
            for a in range(self.game.getActionSize()):
                if moves[a]==0:
                    continue
                nextBoard, _ = self.game.getNextState(board, player, a)
                val = self.minimax(nextBoard, opp, depth-1)
                if best_val is None or val < (best_val, best_move)[0]:
                    (best_val, best_move) = (val, a)
        return (best_val, best_move)[0]
    def alpha_beta(self, board, player, depth, alpha, beta):
        """Find the utility value of the game and the best_val move in the game."""

        if depth == 0:
            return self.eval_fn(board, player)
        #if game is over, we calculate the score
        if self.game.getGameEnded(board, player):
            return self.game.getScore(board, player)

        #get valids
        moves = self.game.getValidMoves(board, player)
        #shuffle the moves in case it places the same position in every game
        #shuffle(moves)
        if moves[-1] == 1:
            return self.eval_fn(board, player)

        opp = -player
        # try each move
        #evaluate max's position and choose the best value
        if player == 1:
            for a in range(self.game.getActionSize()):
                if moves[a]==0:
                    continue
                nextBoard, _ = self.game.getNextState(board, player, a)
                #cut off the branches
                alpha = max(alpha, self.alpha_beta(nextBoard, opp, depth-1, alpha, beta))
                if beta <= alpha: break
            return alpha
        #evaluate min's position and choose the best value
        if player == -1:
            for a in range(self.game.getActionSize()):
                if moves[a]==0:
                    continue
                nextBoard, _ = self.game.getNextState(board, player, a)
                #cut off the branches
                beta = min(beta, self.alpha_beta(nextBoard, opp, depth-1, alpha, beta))
                if beta <= alpha: break
            return beta

    def play(self, board):
        best_val = None
        best_move = None
        player = 1
        moves = self.game.getValidMoves(board, player)

        #shuffle the moves in case it places the same position in every game
        #shuffle(moves)
        if moves[-1] == 1:
            print("pass")
            return  self.boardSize**2#self.eval_fn(board, player) #pass

        opp = -player
        #evaluate max's position and choose the best value
        if player == 1:
            best_val = -INFINITY
            for a in range(self.game.getActionSize()):
                if moves[a]==0:
                    continue
                #print(board)
                nextBoard, _ = self.game.getNextState(board, player, a)
                #alpha = - INFINITY beta = INFINITY
                #we want to choose the max one
                cand_val = max(best_val, self.alpha_beta(nextBoard, opp, 3, -INFINITY, INFINITY))
                if cand_val > best_val:
                    #update best move
                    best_move = a
                    best_val = cand_val
        #evaluate min's position and choose the best value
        if player == -1:
            best_val = INFINITY
            for a in range(self.game.getActionSize()):
                if moves[a]==0:
                    continue
                nextBoard, _ = self.game.getNextState(board, player, a)
                #alpha = - INFINITY beta = INFINITY
                #we want to choose the min one
                cand_val = min(best_val, self.alpha_beta(nextBoard, opp, 3, -INFINITY, INFINITY))
                if cand_val < best_val:
                    #update best move
                    best_move = a
                    best_val = cand_val
        return best_move