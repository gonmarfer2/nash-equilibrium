import sys,os

path = os.path.abspath('../src/')
sys.path.insert(0, path)

from nash_equilibrium import *

if __name__ == '__main__':
    pptls_labels = {
        0:['Piedra','Papel','Tijeras','Lagarto','Spock'],
        1:['Piedra','Papel','Tijeras','Lagarto','Spock']
    }
    n = NashEquilibrium2Players.read_game('./game3',pptls_labels)

    # EQUILIBRIO PURO
    pure_n = n.get_pure_equilibrium()
    pure_n.pretty_print()

    # EQUILIBRIO MIXTO    
    print('Nash Mixed Equilibria:')
    mixed_n = n.get_mixed_equilibrium()
    for player,sol in enumerate(mixed_n):
        print(f'Jugador {player}:')
        sol.pretty_print() 
