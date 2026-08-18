import networkx as nx
def graph_connectivity(Hx, Hz, HX, HZ):
        hgp_graph = nx.Graph()
        node_arr = []
        for i in range(len(Hx[0]) + len(Hx) + len(Hz)):
            node_arr.append(i)
        
        hgp_graph.add_nodes_from(node_arr)
        
        #Add x-q edges

        for i in range(len(Hx)):
            for j in range(len(Hx[i])):
                if (HX[i][j] != 0): hgp_graph.add_edge(j, len(Hx[0])+i)
        
        #Add z-q edges

        for i in range(len(Hz)):
            for j in range(len(Hz[i])):
                if (HZ[i][j] != 0): hgp_graph.add_edge(j, len(Hx[0])+len(Hx)+i)
        
        if (nx.is_connected(hgp_graph)): Tconn = True; 
        else: Tconn = False
        return Tconn