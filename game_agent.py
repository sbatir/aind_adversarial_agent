"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import numpy 
import math #This is legal, right? 


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function! #Dumb baseline 
    if game.is_winner(player):
        return float("inf")
    
    if game.is_loser(player):
        return float("-inf")

    mainMove = len(game.get_legal_moves(player))
    oppMove = len(game.get_legal_moves(game.get_opponent(player)))
    
    emptyMoves = len(game.get_blank_spaces())
    
    #New Algorithm alter policy for swithcing
    if emptyMoves >= 18: #Initiate with aggressive policy. 
        return float(2*mainMove - oppMove)
    else:
        return float(mainMove - 2*oppMove) #Switch to a defensive policy. 
        


    
def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    
    if game.is_winner(player):
        return float("inf")
    
    if game.is_loser(player):
        return float("-inf")

    
    mainMove = len(game.get_legal_moves(player))
    oppMove = len(game.get_legal_moves(game.get_opponent(player)))
    
    return float(mainMove - oppMove)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_winner(player):
        return float("inf")
    
    if game.is_loser(player):
        return float("-inf")
    
    mainMove = len(game.get_legal_moves(player))
    oppMove = len(game.get_legal_moves(game.get_opponent(player)))
    
    return float(2* mainMove - oppMove)
    # This algorithm is an offensive heuristic, it seeks to maximize player movements. 


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Base case move tuple. 
        bestMove = (-1, -1)

        try:
            # Perform a fixed search... using fixed search_depth. 
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Pass if search timeout. 

        # Return the best move from the last completed search iteration
        return bestMove

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout
            
        return self.maxMinMove(game,depth)[0] #return only first element. 
    
    def active_player(self, game):
        return game.active_player == self
        
    def maxMinMove(self,game,depth): #Encapsulate. 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if depth == 0: #Base case
            return (game.get_player_location(self), self.score(game,self))
        #Initialize base case scenarios. 
        bestVal, func, bestMove = None, None, (-1, -1)
        
        #Alternate between max and min layer of tree. 
        if self.active_player(game):
            func, bestVal  = max, float("-inf")
        else:
            func, bestVal = min, float("inf")

        # Iterate through moves, input to next ply. 
        for move in game.get_legal_moves():
            ply = game.forecast_move(move)
            score = self.maxMinMove(ply, depth -1)[1] #py vs next ply. 
            #print(score) #For debugging purposes. 
            if func(bestVal, score) == score:
                bestMove, bestVal = move, score
                
        return (bestMove, bestVal)
        
class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
    
    def active_player(self, game):
        return game.active_player == self

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        
        move = (-1, -1) #base case if no legal moves 
   
        for i in range(1,10000): #Initiate long counter
            try:
                move = self.alphabeta(game, i)  #Prune will return 
            except SearchTimeout:
                break
        return move
        

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")): 
        #Required error check
        #Alpha is lower bound of min layer. 
        #Beta is upper bound of max layer. 
        #Required erro check. 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #Extract the first parameter of the total output of prune, otherwise prune will return tuple of move and score. 
        return self.prune(game, depth)[0] #Harvest the luxury move tuple ONLY. 
        #USE ABOVE IF ENCAPSULATION HELPER FUNCTION IS ONLY WAY. 
        
    def prune(self, game, depth, alpha = float("-inf"), beta = float("inf")):
        #required timer line
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        #Base Case: Embed the depth == 0 segment and default path set to -1,-1
        if depth == 0:
            return (-1, -1), self.score(game, self) 


        #Initialiation 
        value, func, bestMove, alphaBool = None, None, (-1, -1), True
        
        if self.active_player(game):
            value, func, alphaBool = float("-inf"), max, True
        else:
            value, func, alphaBool = float("inf"), min, False
        
        #If no more moves. 
        if not game.get_legal_moves():
            if alphaBool == True:
                return (-1, -1), float("-inf") 
            else:
                return (-1, -1), float("inf")

        #If there exist legal moves, then find the maximizing plaeyr that normallyr eturns to the mvoe with the highest possibl escore. then this move will never propagate up the game tree by the minimizing player, if the move has a score larger than beta. 
        
        #Avoid unecessary upper bound check. 
        minScore, highScore = float("inf"), float("-inf")
        bestMove = (-1, -1)

        if depth ==1: #Recursion - Base Case. 
            if alphaBool:  #If not in alpha
                for move in game.get_legal_moves():
                    nextPly = game.forecast_move(move)
                    currScore = self.score(nextPly, self)
                    # Base case check. 

                    if currScore >= beta:
                        return move, currScore
                    if currScore > highScore:
                        bestMove, highScore = move, currScore
                return bestMove, highScore
            else:
                for move in game.get_legal_moves():
                    nextPly = game.forecast_move(move)
                    currScore = self.score(nextPly, self)
                    if currScore <= alpha:
                        return move, currScore
                    if currScore < minScore:
                        bestMove, minScore = move, currScore
                return bestMove, minScore

        if alphaBool: 
            for move in game.get_legal_moves():
                nextPly = game.forecast_move(move)
                currScore = self.prune(nextPly, depth-1, alpha, beta)[1]
                #If branch yields score better than beta, stop searching further, because that is a lower score 
                if currScore >= beta:
                    return move, currScore
                #Otherwise, update alpha and remember best move. 
                if currScore > highScore:
                    bestMove, highScore = move, currScore
                alpha = max(alpha, highScore)
            return bestMove, highScore
        else:
            for move in game.get_legal_moves():
                nextPly = game.forecast_move(move)
                currScore = self.prune(nextPly, depth-1, alpha, beta)[1]
                #If branch has score worse than alpha, stop searching. 
                if currScore <= alpha:
                    return move, currScore
                #Otherwise, remember the best move and update beta
                if currScore < minScore:
                    bestMove, minScore = move, currScore
                beta = min(beta, minScore)
            return bestMove, minScore
