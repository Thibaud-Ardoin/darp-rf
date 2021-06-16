"""
Internal functions only relevant to the utils.data submodule.
"""

import itertools

def get_node_map(req_keep,nr_original):
    if isinstance(req_keep, int):
        req_keep = range(1, req_keep + 1)

    num_req_keep = len(req_keep)

    Nd = range(1, num_req_keep + 1)

    node_map = dict(zip(req_keep, Nd))
    node_map.update({iold + nr_original: inew + num_req_keep for iold, inew in node_map.items()})
    node_map.update({0: 0, nr_original * 2 + 1: num_req_keep * 2 + 1})

    return node_map, num_req_keep


def euclidean_distance_matrix(posx, posy):
    distance_matrix = dict()
    for i,j in itertools.combinations(posx, 2):
        d = ((posx[i]-posx[j])**2 + (posy[i] - posy[j])**2)**0.5
        distance_matrix[i,j] = d
        distance_matrix[j,i] = d
    for i in posx:
        distance_matrix[i,i] = 0
    return distance_matrix

