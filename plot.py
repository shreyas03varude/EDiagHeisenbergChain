import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def main():

    if len(sys.argv) != 6:
        print('Usage:\n\tpython plot.py <L> <delta> <J1> <J2> <input filename> <output filename>')
        quit()
    L = str(sys.argv[1])
    delta = sys.argv[2]
    J1 = str(sys.argv[3])
    J2 = str(sys.argv[4])
    input = sys.argv[5]
    output = sys.argv[6]

    datap = pd.read_csv(input)
    datap = datap.iloc[:,:-1]
    datap = np.concatenate((datap[10:][::-1], datap), axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.transpose(datap[:, ::-1]),        
                interpolation='none',
                aspect='auto')  # plots with columns reversed

    xticks = np.linspace(0, datap.shape[0] - 1, 5)      
    xlabels = [r'$0$',r'$\frac{\pi}{2}$',r'$\pi$',r'$\frac{3\pi}{2}$',r'$2\pi$']
    yticks = np.linspace(datap.shape[1] - 1, 0, 7)      
    ylabels = np.linspace(0, 3, 7)
    ax.set_xticks(ticks=xticks, labels=xlabels)
    ax.set_yticks(ticks=yticks, labels=ylabels)
    ax.set_title('L = ' + L + ', J1 = ' + J1 + ', J2 = ' + J2 + ', $\\delta$ = ' + delta)
    ax.set_xlabel('q')
    ax.set_ylabel('$\omega$')
    fig.colorbar(im, ax=ax, shrink=0.5)
    plt.savefig(output)

if __name__ == '__main__':
   main()
