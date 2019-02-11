import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

class EXPLORATION():
    """Defined utilities for exploring the data packaged as a dict of patient stats
    and a list of dataframes containing eyetrace data. Eyetrace data is x and y
    motion as well as y motion
    """
    def __init__(self, raw):
        self.raw = raw
        pass

    def plot_N(self, N, bad_times=False):
        ##calc number of rows needed with 2 cols
        ncol = np.math.trunc(np.math.sqrt(N))
        if N > (ncol**2):
            nrow = ncol + 1
        else:
    	    nrow = ncol
        rand_list = [(np.math.trunc(np.random.random()*len(self.raw)) + 1) \
	        for i in range(len(self.raw))]

        fig, ax = plt.subplots(nrow, ncol, figsize=(10,10))

        for idx, a_subplot in enumerate(ax.flatten()):
            x = self.raw.xmotion[rand_list[idx]][0]
            y = self.raw.ymotion[rand_list[idx]][0]
            t = self.raw.timesecs[rand_list[idx]][0]
            a_subplot.scatter(x, y, s=5, c=t,
	                        alpha=0.5, edgecolors='none')
            a_subplot.set_xlabel('x (degrees)', fontsize=20)
            a_subplot.set_ylabel('y (degrees)', fontsize=20)
            title = 'ID' + self.raw.ID[rand_list[idx]]
            a_subplot.set_title(title, fontsize=24)

        plt.tight_layout()
        return fig



def confusion(array):
    """ Plots a confustion matrix as a seaborn heat map
    """
    title = "Confusion Matrix"
    df_cm = pd.DataFrame(array, index = [i for i in ["TRUE","FALSE"]],
              columns = [i for i in ["TRUE","FALSE"]])
    plt.figure(figsize = (10,7))
    sn.set(font_scale=2)
    sn.heatmap(df_cm, annot=True)
    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Prediction')
    return plt

