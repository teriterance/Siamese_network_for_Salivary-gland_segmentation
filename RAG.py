import numpy as np
import matplotlib.pyplot as plt

class Heap:

    def __init__(self):
        """Create Heap data Structure Class"""
        self.Heap = []
        self.Pair = []
        self.Where = []

    def __str__(self) -> str:
        return "Heap: "+str(self.Heap) + "\nPair: " + str(self.Pair) + "\nWhere: " +str(self.Where)
    
    def construct(self, H, E, Where, Pair):
        """Construct class Heap"""
        s, e = int(E/2), E
        self.Where = Where.copy()
        self.Pair = Pair.copy()
        self.Heap = H.copy()
        while s > 1:
            s = s - 1 
            self.shiftdown(s, e)
        return self.Heap

    def shiftdown(self, s, e):
        """Shiftdown configuration"""
        i = s
        j = 2*i

        x, p = self.Heap[i], self.Pair[i]
        while j <= e:
            if j < e :
                if self.Heap[j] > self.Heap[j+1]:
                    j = j + 1
            if x <= self.Heap[j]:
                break
            self.Heap[i] = self.Heap[j]
            pp = self.Pair[j]
            self.Pair[i] = pp
            self.Where[pp] = i
            i, j = j, 2*i

        self.Heap[i], self.Where[p], self.Pair[i] = x, i, p

    def shiftup(self, s):
        """shiftup configuration"""
        i = s
        j = int(i/2)

        x, p = self.Heap[i], self.Pair[i]
        while j >= 1:
            if self.Heap[j] <= x:
                break
            self.Heap[i] = self.Heap[j]
            pp = self.Pair[j]
            self.Pair[i] = pp
            self.Where[pp] = i
            i = j
            j = int(i/2)
        self.Heap[i], self.Where[p], self.Pair[i] = x, i, p

    def update(self, x, s, e):
        """Update function"""
        if x > self.Heap[s]:
            self.Heap[s] = x
            self.shiftdown(s, e)
        elif x < self.Heap[s]:
            self.Heap[s] = x
            self.shiftup(s)
    
    def remove(self, x, s, e):
        """Remove function"""
        x = self.Heap[s]
        self.Heap[s] = self.Heap[e]
        
        p = self.Pair[e]
        self.Pair[s] = p
        self.Where[p] = s

        if x < self.Heap[e]:
            self.shiftdown(s, e-1)
        elif x > self.Heap[e]:
            self.shiftup(s)

    def _empty(self):
        """clear the list"""
        self.Heap = []
        self.Pair = []
        self.Where = []

class Adjency:
    """"""
    def __init__(self, im_shape,im_feature_1, im_feature_2):
        ## chaque pixel va constituer une region
        self.region_list = []
        for i in range(im_shape[0]):
            for j in range(im_shape[1]):
                self.region_list.append([i,j])
        ##construction du graphe
        self.graph = {}
        for a in range(len(self.region_list)):

            if a[0] < 1:
                self.graph[a] = [a[0]+1, a[1]]
                if a[1] < 1: 
                    self.graph[a] = [a[0], a[1]+1]
                elif a[1] + 1 > im_shape[1]:
                    self.graph[a] = [a[0], a[1]-1]

            elif a[0] + 1 > im_shape[0]:
                self.graph[a] = [a[0]-1, a[1]]
                if a[1] < 1: 
                    self.graph[a] = [a[0], a[1]+1]
                elif a[1] + 1 > im_shape[1]:
                    self.graph[a] = [a[0], a[1]-1]

            elif a[0] > 1 and a[0] < im_shape[0] -1 and a[1] > 1 and a[1] < im_shape[1] - 1:
                self.graph[a] = [a[0]-1, a[1]]
                self.graph[a] = [a[0]+1, a[1]]
                self.graph[a] = [a[0], a[1]-1]
                self.graph[a] = [a[0], a[1]+1]
    
    def disimilarity_criterion(self):
        """"""
        pass
    

    def disimilarity_criterion_ward_method(self):
        """"""
        pass



def RAG_val(img):
    """RAG function"""

    #Initialization
    #each pixel is region

    pass

if __name__ == "__main__":
    h = Heap()
    print(h)
    print("\n")
    H = [1, 2, 3, 4]
    P = [1, 2, 0, 3]#, [1,3]]
    W = [1,0,2, 4]
    h.construct(H, 4, W, P)
    print(h)
    print("\n")
    h.update(2, 3, 3)
    print(h)
    print("\n")
    h.remove(2, 3, 3)
    print(h)
    print("\n")