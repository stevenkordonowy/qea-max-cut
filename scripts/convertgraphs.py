import csv
import networkx as nx

# Read a CSV into a networkx graph
# Format matches samples from https://web.stanford.edu/~yyye/yyye/Gset/
def read_graph(graph_name):
    G = nx.Graph()
    with open('scripts/data/{0}.txt'.format(graph_name), 'r') as csvfile:
        graph = csv.reader(csvfile, delimiter=' ')
        _headers = next(graph, None)
        for edge in graph:
            # We want the nodes of our graph to start at 0
            G.add_edge(int(edge[0]) - 1, int(edge[1]) - 1, weight=float(edge[2]))

    return G

# Read a CSV into a networkx graph
# Format matches samples from https://web.stanford.edu/~yyye/yyye/Gset/
def read_graph_into_array(graph_name):
    G = read_graph(graph_name)
    return G.adj