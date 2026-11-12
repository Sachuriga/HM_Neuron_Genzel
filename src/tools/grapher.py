# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:37:54 2021

@author: User
"""

import networkx as nx
from matplotlib import pyplot as plt
try:                              # (there is no 'utils' package: mask.py is in src/tools)
    from tools import mask        # src/ on the path
except ImportError:
    import mask                   # run from src/tools
import csv

from tkinter import Tk, filedialog, Label
import numpy as np
import logging
import argparse
import os

def maze_graph(nodelist):
    flower_graph = {1: [2, 7],
                2: [1, 3],
                3: [2, 4, 9],
                4: [3, 5],
                5: [4, 11],
                6: [7, 13],
                7: [6, 1, 8],
                8: [7, 9, 15],
                9: [3, 8, 10],
                10: [9, 11, 17],
                11: [5, 10, 12],
                12: [11, 19],
                13: [6, 14],
                14: [13, 15, 20],
                15: [8, 14, 16],
                16: [15, 17, 22],
                17: [10, 16, 18],
                18: [17, 19, 24],
                19: [12, 18],
                20: [14, 21],
                21: [20, 22],
                22: [16, 21, 23],
                23: [22, 24],
                24: [18, 23]}

    island_prefixes = ['1', '2', '3', '4']

    bridge_edges = [('124', '201', 60),
                ('302', '121', 172),
                ('223', '404', 169),
                ('324', '401', 60),
                ('305', '220', 60)]

    bridge_edges_uw = [('124', '201'),
                ('121', '302'),
                ('223', '404'),
                ('324', '401'),
                ('305', '220')]

    graph_prototype= {}

    for letter in island_prefixes:
        for node_suffix, edges in flower_graph.items():
            if node_suffix < 10:
                first_point = letter +'{}{}'.format(0, str(node_suffix))
            else:
                first_point = letter +'{}'.format(node_suffix)
            edge_list = []
            for n in edges:
                if n < 10:
                    second_point = letter + '{}{}'.format(0, str(n))
                else:
                    second_point = letter + '{}'.format(n)
                edge_list.append(second_point)
            graph_prototype[first_point] = edge_list

    mg = nx.Graph()
    xg = nx.Graph()
    mg = nx.from_dict_of_lists(graph_prototype)
    xg = nx.from_dict_of_lists(graph_prototype)
    for e in mg.edges():
        mg[e[0]][e[1]]['weight'] = 30

    mg.add_weighted_edges_from(bridge_edges)
    xg.add_edges_from(bridge_edges_uw)
    simple_path = dict(nx.all_pairs_shortest_path(xg))
    dijkstra_path = dict(nx.all_pairs_dijkstra_path(mg, weight = 'weight'))

    return mg, simple_path, dijkstra_path



def path_graph(inputfile):
    pg = nx.DiGraph()
    node_list = []

    with open(inputfile) as file:
        lines = file.read().splitlines(keepends = False)
        for line in lines:
            for node in line.split(','):
                pg.add_node(node)
                node_list.append(node)
            break
        for num, edge_points in enumerate(node_list):
            if num:
                pg.add_edge(node_list[num - 1], node_list[num])
            print('nodelist', node_list)
    return pg, node_list


#find paths where shortest path by node is lesser than shortest path by distance
def find_shortest_path(nodelist, simple_path, dijkstra_path):
    reg_nodes = []
    count = 0

    with open(nodelist, 'r') as nl:
        read_nodes = csv.reader(nl)
        for nodes in read_nodes:
            reg_nodes.append(nodes)
        for fpoint in reg_nodes:
            for spoint in reg_nodes:
                x1 = fpoint[0]
                x2 = spoint[0]
                sp = simple_path[x1][x2]
                dp = dijkstra_path[x1][x2]
                if len(simple_path[x1][x2]) < len(dijkstra_path[x1][x2]):

                    with open('shortest_file_paths.txt', 'a+') as file:
                        file.write(f'{x1} {x2}')
                        file.write('\n')
                        file.write(f'{len(sp)} : ')
                        file.writelines("%s" % node for node in sp)
                        file.write('\n')
                        file.write(f'{len(dp)} : ')
                        file.writelines("%s" % node for node in dp)
                        file.write('\n')

                    print('sp:{}, {} {}'.format(len(sp), sp[0], sp[-1]))
                    print('dp:{}'.format(len(dp)))
                    count += 1
                else:
                    continue
    print(count)

if __name__ == "__main__":
    inputfile = '2021-03-08_Rat1stitched.txt'
    nodelist = 'new_list.csv'
    #enter inputfile to be graphed
    #enter = input('Enter unique file name: ')
    #inputfile = '' if not enter else enter
    
    
    path_graph(inputfile)
    nodes_dict = mask.create_node_dict(nodelist)
    G, simple_path, dijkstra_path = maze_graph(nodelist)
    find_shortest_path(nodelist, simple_path, dijkstra_path)

    N = nx.Graph()
    O = nx.Graph()

    with plt.xkcd():
        nx.draw(G, pos = nodes_dict)
        nx.draw_networkx_labels(G, pos = nodes_dict)

        #nx.draw_networkx_edges(H, nodes_dict, width = 2, edge_color = 'orange')
        #nx.draw_networkx_nodes(H, nodes_dict, nodelist = [nl[0]], node_color = 'g')
        #nx.draw_networkx_nodes(H, nodes_dict, nodelist = [nl[-1]], node_color = 'r')
    plt.show()


