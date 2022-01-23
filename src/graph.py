from .utility import *

import matplotlib.pyplot as plt
import seaborn as sns

def violin(df, plot_var, fig_size=6, fig_ratio=(1,1), fig_name=None, dpi=500):
    for i, var_name in enumerate(plot_var):
        plt.figure(figsize=(fig_size*fig_ratio[0],fig_size*fig_ratio[1]))
        sns.violinplot(data = df[var_name])
        plt.xlabel("distribution")
        plt.ylabel(f"{var_name}")
        
        if fig_name:
            createfolder("./graph/violin")
            plt.savefig(f"./graph/violin/{fig_name}_{var_name}_{fig_size}_{dpi}dpi.png", dpi=dpi)