import numpy as np
import scipy as sp
import os, pandas, re
from fractions import Fraction

def calculate_result(path):
    win = None
    pathlist = [int(p) for p in list(path)]
    turns = [i%2 for i in range(len(pathlist))]
    tabletop = np.array([[-1 for col in range(3)] for row in range(3)])
    for action,player in zip(pathlist,turns):
        tabletop[action//3,action%3] = player
        win_rows = check_rows(tabletop)
        win_cols = check_cols(tabletop)
        win_diagonals = check_diagonals(tabletop)
        if win_rows != None:
            win = win_rows
        elif win_cols != None:
            win = win_cols
        elif win_diagonals != None:
            win = win_diagonals
    return win
        

def check_rows(tabletop):
    '''Devuelve None si no hay ganador, o el número del jugador con 3 en raya'''
    win = None
    for row in tabletop:
        for i in range(2):
            if np.all(row == i):
                win = i
    return win

def check_cols(tabletop):
    '''Devuelve None si no hay ganador, o el número del jugador con 3 en raya'''
    win = None
    for col in tabletop.T:
        for i in range(2):
            if np.all(col == i):
                win = i
    return win

def check_diagonals(tabletop):
    '''Devuelve None si no hay ganador, o el número del jugador con 3 en raya'''
    win = None
    MAIN_DIAGONAL = tabletop.take([0,4,8])
    SECOND_DIAGONAL = tabletop.take([2,4,6])
    for p in range(2):
        if np.all(MAIN_DIAGONAL == p):
            win = p
        elif np.all(SECOND_DIAGONAL == p):
            win = p
    return win


class NashEquilibrium2Players():
    '''
    game: list of matrices with the payoff function for each player
    '''
    def __init__(self,game_list,actions) -> None:
        assert isinstance(game_list,list), f"A game must be of list of ndarray types, but got {type(game_list)}"
        assert len(game_list) == 2, f"There are {len(game_list)} matrices although there should only be 2"
        assert all([isinstance(matrix,np.ndarray) for matrix in game_list]), f"A game list must only contain ndarray type objects"
        assert len(actions) == 2, f"There are {len(actions)} action lists where there should be only 2"
        self.game = game_list
        self.actions = actions

    @staticmethod
    def read_game(folder):
        '''
        Reads the csv files in a folder with the format
        0player...
        1player...
        '''
        assert os.path.exists(folder), "The input folder does not exist"
        game_list = []
        for root,dirs,files in os.walk(folder):
            for file in files:
                if re.match('[0-1]player.*',file):
                    matrix = pandas.read_csv(f'{root}/{file}',header=None)
                    matrix_np = matrix.to_numpy()
                    game_list.append(matrix_np)
        assert len(game_list) > 1, "There are not enough payoff matrices"
        assert len(game_list) < 3, "There are too many payoff matrices"
        game_0_shape = game_list[0].shape
        assert all([game.shape == game_0_shape for game in game_list]), "The payoff matrices must have the same shape"
        default_labels = {i:[j for j in range(game_0_shape[i])] for i in range(2)}
        return NashEquilibrium2Players(game_list,default_labels)
    
    def __str__(self) -> str:
        res = ''
        for player in range(2):
            res += f'Player {player}: \n {str(self.game[player])} \n\n'
        return res
    
    def __repr__(self) -> str:
        return f'Game {hash(self.game)}'
    
    def get_actions_it(self,player):
        return list(range(len(self.actions[player])))
        
    def get_payoff(self,player,strategy):
        return self.game[strategy[0],strategy[1]][player]
    
    def get_payoffs_by_other_actions(self,player):
        if player == 0:
            return {action:self.game[player].transpose()[action] for action in self.get_actions_it(1)}
        elif player == 1:
            return {action:self.game[player][action] for action in self.get_actions_it(0)}

    def get_non_dominated_strategies(self):
        #Player 0
        non_dominated_0 = []
        for i in range(len(self.game[0])):
            for j in range(1,len(self.game[0])):
                if any(self.game[0][i] - self.game[0][j] > 0):
                    non_dominated_0.append(i)
                    break

        non_dominated_1 = []
        game_transpose = self.game[1].transpose()
        for i in range(len(game_transpose)):
            for j in range(1,len(game_transpose)):
                if any(game_transpose[i] - game_transpose[j] > 0):
                    non_dominated_1.append(i)
                    break

        new_game = []
        for game in self.game:
            process_game = np.take(game,non_dominated_0,axis=0)
            new_game.append(np.take(process_game,non_dominated_1,axis=1))
        labels = {
            0:np.take(self.actions[0],non_dominated_0),
            1:np.take(self.actions[1],non_dominated_1)
        }
        return NashEquilibrium2Players(new_game,labels)
    
    def get_pure_equilibrium(self):
        max_strats = {i:set() for i in range(2)}
        '''Para cada jugador, quiero coger las casillas que marca con mejor
        payoff y hacer la intersección con el resto de jugadores. Donde
        coinciden, es la estrategia de equilibrio'''
        for player in range(2):
            for other_action,payoff in self.get_payoffs_by_other_actions(player).items():
                best_actions = np.argwhere(payoff == np.amax(payoff))[0]
                if player == 0:
                    max_strats[player] = max_strats[player].union(set(tuple(tuple([a,other_action]) for a in best_actions)))
                elif player == 1:
                    max_strats[player] = max_strats[player].union(set(tuple(tuple([other_action,a]) for a in best_actions)))
        
        res = max_strats[0].intersection(max_strats[1]) 
        return res
    
    def get_mixed_equilibrium(self):
        pass

#def calculate_percentages

if __name__ == '__main__':
    #print(calculate_result('012345678'))
    '''mat = np.array([[2,-1,-1],[1,-2,-1],[1,1,1]])
    print(mat)
    b = np.array([0,0,1])
    sol = sp.linalg.solve(mat,b)
    print([str(Fraction(x).limit_denominator()) for x in sol])'''
    n = NashEquilibrium2Players.read_game('../test/game1')
    #print(n.get_pure_equilibrium())
    print(n.get_non_dominated_strategies().get_pure_equilibrium())