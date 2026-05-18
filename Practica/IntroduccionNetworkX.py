import networkx as nx
import matplotlib.pyplot as plt

# grafica simple
G=nx.Graph()
#G=nx.DiGraph() # grafica dirigida
#agregar nodos
G.add_node(1)
G.add_node(2)
G.add_nodes_from([3,4,5])

#agregar aristas
G.add_edge(1,2) # Arista de nodo 1 a nodo 2
G.add_edge(2,3) # Arista de nodo 2 a nodo 3
G.add_edges_from([(3,4),(4,5),(5,1)]) # Agregar varias aristas a la vez


nx.draw(G, with_labels=True)
plt.show()
