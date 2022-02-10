from src.utility import *

from re import sub
import matplotlib.pyplot as plt

from cv2 import kmeans
from sklearn.cluster import KMeans
from collections import Counter

class KmeanCluster():
    
    def __init__(self,
                 df,
                 target_var):
        
        self.df = df.copy()
        self.target_var = target_var
        self.num_cluster = 0

        target_data = df[target_var]
        mean = target_data.mean(axis=0) + 1e-20
        std  = target_data.std(axis=0)  + 1e-20
        self.df_scaled = (target_data - mean)/std
        
    def test_inertia(self, min_cluster=1, max_cluster=10):
        inertia_list = []
        
        for i in range(min_cluster, max_cluster+1):
            model = KMeans(n_clusters=i)
            model.fit(self.df_scaled)
            inertia_list.append(model.inertia_)
        
        plt.figure(figsize=(12,6))
        plt.plot(range(min_cluster, max_cluster+1), inertia_list, 'bo-')
        plt.xlabel("Number of Clusters(k)")
        plt.ylabel("Inertia")
        
        self.inertia = inertia_list
    
    def model(self, num_cluster, random_state=42):
        self.num_cluster = num_cluster
        
        kmean_model = KMeans(n_clusters=num_cluster, random_state=random_state)
        kmean_model.fit(self.df_scaled)
        
        cluster_label = kmean_model.labels_
        self.df['cluster'] = cluster_label
        cluster_rank = sorted(Counter(cluster_label).items(), 
                             reverse = True, key = lambda x: x[1])
        cluster_list = [x[0] for x in cluster_rank]
        
        color_label = []
        for c in self.df['cluster']:
            for i, cluster in enumerate(cluster_list):
                if c == cluster:
                    color_label.append(f"C{i+1}")
        self.df['color'] = color_label
        return kmean_model
    
    def plot(self, plot_var, 
                   fig_size=6, fig_ratio=(1,1), s=10, alpha=0.8,
                   subplot=None, fig_name=None, dpi=500, save_fig=True):
        if self.num_cluster == 0:
            print("Build a cluster model first")
            
        elif subplot == None:
            color_list = self.df['color'].unique()
            for var_name in plot_var:
                plt.figure(figsize=(fig_size*fig_ratio[0],fig_size*fig_ratio[1]))
                for color in color_list:
                    plot_data = self.df[self.df['color']==color]
                    plt.scatter(plot_data.index, plot_data[var_name], 
                                color=color, s=s, alpha=alpha)
                plt.title(var_name)
                plt.show()
                
                if fig_name and save_fig:
                    createfolder("./graph/clustering")
                    plt.savefig(f"./graph/clustering/{fig_name}_{dpi}dpi", dpi=dpi)
        
        else:
            color_list = self.df['color'].unique()
            plt.figure(figsize=(fig_size*fig_ratio[0]*subplot[1],
                                fig_size*fig_ratio[1]*subplot[0]))
            for i, var_name in enumerate(plot_var):
                plt.subplot(subplot[0], subplot[1], i+1)
                for color in color_list:
                    plot_data = self.df[self.df['color']==color]
                    plt.scatter(plot_data.index, plot_data[var_name],
                                color=color, s=s, alpha=alpha)
                plt.title(var_name)
            plt.tight_layout()
            plt.show()
            if fig_name:
                createfolder("./graph/clustering")
                plt.savefig(f"./graph/clustering/{fig_name}_{dpi}dpi", dpi=dpi)