from skimage.future.graph import rag
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np



def rag(img, out1, out2):
    size_x, size_y = img.shape
    g = rag.RAG()
    for i in range(16, size_x - 16):
        for j in range(16, size_y - 16):
            g.add_edge((i,j), (i,j + 1), weight = out1[i, j] + out2[i, j+1] )
            g.add_edge((i,j), (i,j-1), weight = out1[i, j] + out2[i, j-1] )
            g.add_edge((i,j), ((i-1),j), weight = out1[i, j] + out2[i-1, j] )
            g.add_edge((i,j), ((i+1),j), weight = out1[i, j] + out2[i+1, j] )
    
    for n in g.nodes():
        g.nodes[n]['labels'] = [n]

        
    flag = False
    while flag == False:
        pass


def max_edge(g, src, dst, n):
    """Callback to handle merging nodes by choosing maximum weight.

    Returns a dictionary with `"weight"` set as either the weight between
    (`src`, `n`) or (`dst`, `n`) in `g` or the maximum of the two when
    both exist.

    Parameters
    ----------
    g : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `g` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dict with the "weight" attribute set the weight between
        (`src`, `n`) or (`dst`, `n`) in `g` or the maximum of the two when
        both exist.
    """

    w1 = g[n].get(src, {'weight': -np.inf})['weight']
    w2 = g[n].get(dst, {'weight': -np.inf})['weight']
    return {'weight': max(w1, w2)}


def display(g, title):
    """Displays a graph with the given title."""
    pos = nx.circular_layout(g)
    plt.figure()
    plt.title(title)
    nx.draw(g, pos)
    nx.draw_networkx_edge_labels(g, pos, font_size=20)

gc = g.copy()

display(g, "Original Graph")

g.merge_nodes(1, 3)
display(g, "Merged with default (min)")

gc.merge_nodes(1, 3, weight_func=max_edge, in_place=False)
display(gc, "Merged with max without in_place")

plt.show()