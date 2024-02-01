import sys,os

path = os.path.abspath('../src/')
sys.path.insert(0, path)

from nash_equilibrium import *

if __name__ == '__main__':
    labels = {
        0:['Confiesa','No confiesa'],
        1:['Confiesa','No confiesa']
    }
    n = NashEquilibrium2Players.read_game('./game1',labels)

    # EQUILIBRIO PURO
    pure_n = n.get_pure_equilibrium()
    pure_n.pretty_print()

    # EQUILIBRIO MIXTO
    print('Nash Mixed Equilibria:')
    mixed_n = n.get_mixed_equilibrium()
    for player,sol in enumerate(mixed_n):
        print(f'Jugador {player}:')
        sol.pretty_print() 
