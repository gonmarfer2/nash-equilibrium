import numpy as np
import scipy as sp
import os, pandas, re
from fractions import Fraction

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
    def read_game(folder,labels=None):
        '''
        Reads the csv files in a folder with the format
        0player...
        1player...
        Introduce a dict with labels for each player if needed
        '''
        assert os.path.exists(folder), "The input folder does not exist"
        assert labels == None or (isinstance(labels,dict) and len(labels) == 2), "Labels must be a dict of length 2"
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
        if labels:
            default_labels = labels
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
    
    def get_payoff_matrix(self,player):
        return self.game[player]
    
    def get_payoffs_by_other_actions(self,player):
        payoff_matrix = self.game[player]
        if player == 0:
            return {action:payoff_matrix.transpose()[action] for action in self.get_actions_it(1)}
        elif player == 1:
            return {action:payoff_matrix[action] for action in self.get_actions_it(0)}

    def get_non_dominated_strategies(self):
        #Player 0
        non_dominated_0 = []
        for i in range(len(self.game[0])):
            for j in range(len(self.game[0])):
                if i != j and any(self.game[0][i] - self.game[0][j] >= 0):
                    non_dominated_0.append(i)
                    break

        non_dominated_1 = []
        game_transpose = self.game[1].transpose()
        for i in range(len(game_transpose)):
            for j in range(len(game_transpose)):
                if i != j and any(game_transpose[i] - game_transpose[j] >= 0):
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
        non_dom_game = self.get_non_dominated_strategies()
        max_strats = {i:set() for i in range(2)}
        '''Para cada jugador, quiero coger las casillas que marca con mejor
        payoff y hacer la intersección con el resto de jugadores. Donde
        coinciden, es la estrategia de equilibrio'''
        for player in range(2):
            for other_action,payoff in non_dom_game.get_payoffs_by_other_actions(player).items():
                best_actions = list(np.argwhere(payoff == np.amax(payoff)).T[0])
                if player == 0:
                    max_strats[player] = max_strats[player].union(set(tuple(tuple([a,other_action]) for a in best_actions)))
                elif player == 1:
                    max_strats[player] = max_strats[player].union(set(tuple(tuple([other_action,a]) for a in best_actions)))
        
        res = max_strats[0].intersection(max_strats[1]) 
        return PureNashEquilibrium(res,self.actions)
    
    def get_mixed_equilibrium(self):
        # Coger la matriz de payoff de un jugador:
        #   Si es jugador filas: tomar la primera columna, restarla al resto y añadir una columna 1,1,1,...
        #   Si es jugador columnas: tomar la primera fila, restarla al resto y añadir una fila 1,1,1,...
        # JUGADOR FILAS
        non_dom_game = self.get_non_dominated_strategies()
        payoff0 = non_dom_game.get_payoff_matrix(0)
        first_col = payoff0[0]
        remove_first_row = (payoff0 - first_col)[1:]
        sum0 = np.zeros(shape=(1,remove_first_row.shape[0]))
        mixed_strat_matrix0 = np.concatenate((remove_first_row,
                                             sum0.T),
                                             axis=1)
        mixed_strat_matrix0 = np.concatenate((mixed_strat_matrix0,
                                             np.array([[1 for i in range(mixed_strat_matrix0.shape[1])]])),
                                             axis=0)
        try:
            sol0 = sp.linalg.solve(mixed_strat_matrix0[:,:-1],mixed_strat_matrix0[:,-1])
        except sp.linalg.LinAlgError:
            sol0 = None

        payoff1 = non_dom_game.get_payoff_matrix(1).transpose()
        first_row = payoff1[0]
        remove_first_col = (payoff1 - first_row)[1:]
        sum1 = np.zeros(shape=(1,remove_first_col.shape[0]))
        mixed_strat_matrix1 = np.concatenate((remove_first_col,
                                             sum1.T),
                                             axis=1)
        mixed_strat_matrix1 = np.concatenate((mixed_strat_matrix1,
                                             np.array([[1 for i in range(mixed_strat_matrix1.shape[1])]])),
                                             axis=0)
        
        try:
            sol1 = sp.linalg.solve(mixed_strat_matrix1[:,:-1],mixed_strat_matrix1[:,-1])
        except sp.linalg.LinAlgError:
            sol1 = None

        return (MixedNashEquilibrium(tuple(x for x in sol0) if sol0 is not None else None,non_dom_game.actions[0],self.actions[0]),
                MixedNashEquilibrium(tuple(x for x in sol1) if sol1 is not None else None,non_dom_game.actions[1],self.actions[1]))

class PureNashEquilibrium():
    def __init__(self,cell_list,actions) -> None:
        self.strategy = cell_list
        self.actions = actions

    def __str__(self) -> str:
        return str([(self.actions[0][i],self.actions[1][j]) for i,j in self.strategy])
    
    def __repr__(self) -> str:
        return str(self)
    
    def pretty_print(self):
        if len(self.strategy) == 0:
            print('There are no pure Nash Equilibria\n')
        else:
            res = 'Nash Pure Equilibria:\n'
            for strategy in [(self.actions[0][i],self.actions[1][j]) for i,j in self.strategy]:
                res += f'\t({strategy[0]},{strategy[1]})\n'
            print(res)

class MixedNashEquilibrium():
    def __init__(self,strategy,positive_actions,total_actions) -> None:
        assert isinstance(strategy,tuple) or strategy==None
        strategy_dict = {}
        if strategy:
            for s,action in zip(strategy,positive_actions):
                strategy_dict[action] = s
            for action in total_actions:
                if action not in strategy_dict:
                    strategy_dict[action] = 0
        self.strategy = strategy_dict

    def __str__(self) -> str:
        return str(tuple(f'{a}={Fraction(s).limit_denominator()}' for a,s in self.strategy.items()))
    
    def __repr__(self) -> str:
        return str(self)
    
    def pretty_print(self):
        if self.strategy:
            res = ''
            for a,s in self.strategy.items():
                res += f'\t{a} = {Fraction(s).limit_denominator()}\n'
            print(res)
        else:
            print('\tThere are infinite mixed strategies for this player.\n')


if __name__ == '__main__':
    pptls_labels = {
        0:['Piedra','Papel','Tijeras','Lagarto','Spock'],
        1:['Piedra','Papel','Tijeras','Lagarto','Spock']
    }
    labels = {
        0:['Adelanta','Continúa'],
        1:['Adelanta','Continúa']
    }
    n = NashEquilibrium2Players.read_game('./test/game1',labels)
    mixed_n = n.get_mixed_equilibrium()
    for player,sol in enumerate(mixed_n):
        print(f'Jugador {player}:')
        sol.pretty_print() 

    n.get_pure_equilibrium().pretty_print()