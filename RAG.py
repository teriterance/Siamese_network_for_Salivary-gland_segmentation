import numpy as np
import matplotlib.pyplot as plt

class Heap:

    def __init__(self):
        """Create Heap data Structure Class"""
        self.Heap = []
        self.Pair = []
        self.Where = []
    
    def construct(self, E, Where, Pair):
        """Construct class Heap"""
        s, e = int(E/2), E
        self.Where = Where.copy()
        self.Pair = Pair.copy()
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
        j = 2*i

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
        if x > self.Heap(s):
            self.Heap(s) = x
            self.shiftdown(s, e)
        elif x < self.Heap(s):
            self.Heap(s) = x
            self.shiftup(s)
    
    def remove(self, x, s, e):
        x = self.Heap(s)
        self.Heap(s) = self.Heap(e)
        
        p = self.Pair[e]
        self.Pair[s] = p
        self.Where[p] = s

        if x < self.Heap[e]:
            self.shiftdown(s, e-1)
        elif x > self.Heap[e]:
            self.shiftup()

    def empty(self):
        self.Heap = []
        self.Pair = []
        self.Where = []

def rag(img):
    pass

if __name__ == "__main__":
    pass