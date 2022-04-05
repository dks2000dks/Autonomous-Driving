import numpy as np
import matplotlib.pyplot as plt

class KMeans():
    def __init__(self,k):
        self.k = k
        self.DMeasure = []
    
    def InitiliseCentroids(self,X):
        # Assigning Random k-points as Clusters
        N = X.shape[1]
        Ind = np.random.randint(N,size=(self.k,))
        G = []
        for i in Ind:
            G.append(X[:,i])
            
        return np.transpose(G)
    
    def DistortionMeasure(self,r,X,Centroids):
        # Calculating Distortion Measure
        """
        Distances: Distance of Each Data Point from each Centriod. Shape:(n,k)
        """
        Distances = np.linalg.norm((np.repeat(np.expand_dims(X,axis=1),self.k,axis=1)- np.expand_dims(Centroids,axis=-1)), axis=0)
        J = np.sum(np.multiply(Distances,r))
        
        return J
    
    def ClusterData(self,X,Centroids):
        # Assigning Cluster to Each Point in X
        N = X.shape[1]
        r = np.zeros((self.k,N))
        
        Distances = np.linalg.norm((np.repeat(np.expand_dims(X,axis=1),self.k,axis=1)- np.expand_dims(Centroids,axis=-1)), axis=0)
        Clusters = np.argmin(Distances,axis=0)
        
        # Representing Clusters as 1-of-K Representation
        r = np.transpose(np.eye(self.k)[Clusters])
            
        return r
    
    def CalculateCentroids(self,X,r):
        # Calculating Centroids
        d = X.shape[0]
        Centroids = np.zeros((d,self.k))
        
        Num = np.matmul(X,np.transpose(r))
        Den = np.sum(r,axis=1)
        Centroids = np.divide(Num,np.repeat(np.expand_dims(Den,axis=0),d,axis=0))
        return Centroids
        
        
    def fit(self,X,Epsilon):
        # Iterating over all Data Points
        Centroids = self.InitiliseCentroids(X)
        Loss = []
        e = 0
        
        while 1:
            r = self.ClusterData(X,Centroids)
            J = self.DistortionMeasure(r,X,Centroids)
            NCentroids = self.CalculateCentroids(X,r)
            Diff = np.sum(np.linalg.norm(np.abs(NCentroids-Centroids),axis=1))
            #print ("Epoch/Iteration: %-*s  Error in Centroids: %-*s  Distortion Measure: %s" % (6,e,24,Diff,J))
            Loss.append(Diff)
            e += 1
            if Diff < Epsilon:
                break
                
            Centroids = NCentroids

        return Loss,np.argmax(r,axis=0),Centroids
                
    def LossPlot(self,Loss):
        # Plotting Loss
        """
        Loss: Loss Data/ Distortion Measures before Each Iteration
        """
        Loss = np.array(Loss)
        
        plt.figure(figsize=(8,6))
        plt.plot(np.arange(Loss.shape[0]),Loss)
        plt.grid()
        plt.title("Loss")
        plt.xlabel('Epochs/Iterations')
        plt.ylabel('Error in Centroids')
        plt.show()
        
    def PlotClusters(self,X,y):
        # Plotting Clusters
        """
        y: Labels of Data Points
        """
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        plt.title("Clustering Data")
        ax.scatter(X[0],X[1],X[2],c=y,cmap='jet')
        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")
        plt.ioff()
        plt.show()
        
    def ClusteredOutput(self,X,Labels,Centroids):
        # Returning Data after assigning to Clusters
        """
        Labels: Labels of Data Points
        """
        Y = np.zeros(X.shape)

        for i in range(X.shape[1]):
            Y[:,i] = Centroids[:,Labels[i]]

        return Y