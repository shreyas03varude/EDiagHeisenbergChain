import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def main():
    L = sys.argv[1]
    for ratio in np.linspace(0, 1.0, 11, dtype=float):
        datap = pd.read_csv("datas/" + str(L) + "/L=" + str(L) + " and ratio = " + str(ratio)[:3] + ".csv", header=None).values
        datap = datap[:,:-1]
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
        ax.set_title('L = ' + str(L) + ', J2/J1 = ' + str(ratio)[:3])
        ax.set_xlabel('q')
        ax.set_ylabel('$\omega$')
        # ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
        # ax.xaxis.set_major_locator(tck.MultipleLocator(base=0.5))
        fig.colorbar(im, ax=ax, shrink=0.5)
        plt.savefig("plots/" + str(L) + "/plot for L = " + str(L) + " and ratio = " + str(ratio)[:3] + ".png")

if __name__ == "__main__":
   main()
