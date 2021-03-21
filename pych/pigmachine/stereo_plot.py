"""Class to improve the stereo plot function"""

import matplotlib.pyplot as plt


class StereoPlot():
    lon0 = -100.875
    def __init__(self, nrows=1, ncols=1,
                 figsize=(12,10), background='black'):

        fig,axs = plt.subplots(nrows=nrows,ncols=ncols,
                               figsize=figsize,
