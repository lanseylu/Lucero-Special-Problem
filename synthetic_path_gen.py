import random
from typing import List, Dict, Optional
from covid19_supermarket_abm.utils.create_store_network import create_store_network
import networkx as nx
import numpy as np
from covid19_supermarket_abm.utils.create_synthetic_baskets import get_all_shortest_path_dicts
from covid19_supermarket_abm.utils.load_example_data import load_example_store_graph, load_example_paths
import json

"""The synthetic path generator generates a random customer path as follows:
    First, it samples the size K of the shopping basket using a log-normal random variable with parameter mu and sigma.
    Second, it chooses a random entrance node as the first node v_1 in the path.
    Third, it samples K random item nodes, chosen uniformly at random with replacement from item_nodes, which we denote by
    v_2, ... v_K+1.
    Fourth, it samples a random till node and exit node, which we denote by v_K+2 and v_K+3.
    The sequence v_1, ..., v_K+3 is a node sequence where the customer bought items, along the the entrance, till and exit
    nodes that they visited.
    Finally, we convert this sequence to a full path on the network using the shortest paths between consecutive nodes
    in the sequence.
    We use shortest_path_dict for this.
    For more information, see the Data section in https://arxiv.org/pdf/2010.07868.pdf

    The batch_size specifies how many paths we generate in each batch (for efficiency reasons).
    """

pos = {
    0: (12, -4), 1: (12, 0), 2: (12, 4), 3: (12, 8), 4: (12, 12), 5: (15, 12), 6: (18, 12), 7: (22, 12),
    8: (26, 12), 9: (30, 12), 10: (54, 7), 11: (30, 7), 12: (33, 13), 13: (37, 13), 14: (41, 13),
    15: (45, 13), 16: (50, 13), 17: (54, 13), 18: (58, 13), 19: (62, 13), 20: (66, 13), 21: (70, 13),
    22: (73, 13), 23: (76, 13), 24: (12, 17), 25: (18, 17), 26: (22, 17), 27: (26, 17), 28: (30, 17),
    29: (33, 17), 30: (37, 17), 31: (41, 17), 32: (45, 17), 33: (50, 17), 34: (54, 17), 35: (58, 17),
    36: (62, 17), 37: (66, 17), 38: (70, 17), 39: (73, 17), 40: (76, 17), 41: (82.5, 17), 42: (12, 21),
    43: (18, 21), 44: (22, 21), 45: (26, 21), 46: (30, 21), 47: (33, 21), 48: (37, 21), 49: (41, 21),
    50: (45, 21), 51: (50, 21), 52: (54, 21), 53: (58, 21), 54: (62, 21), 55: (66, 21), 56: (70, 21),
    57: (76, 21), 58: (78, 21), 59: (81, 21), 60: (84, 21), 61: (87, 21), 62: (12, 26), 63: (18, 26),
    64: (22, 26), 65: (26, 26), 66: (30, 26), 67: (33, 26), 68: (37, 26), 69: (41, 26), 70: (45, 26),
    71: (50, 26), 72: (54, 26), 73: (58, 26), 74: (62, 26), 75: (66, 26), 76: (70, 26), 77: (74, 26),
    78: (78, 26), 79: (81, 26), 80: (84, 26), 81: (87, 26), 82: (12, 32), 83: (18, 32), 84: (22, 32),
    85: (26, 32), 86: (30, 32), 87: (33, 32), 88: (37, 32), 89: (41, 32), 90: (45, 32), 91: (50, 32),
    92: (54, 32), 93: (58, 32), 94: (62, 32), 95: (66, 32), 96: (70, 32), 97: (74, 32), 98: (78, 32),
    99: (81, 32), 100: (84, 32), 101: (87, 32), 102: (12, 36), 103: (18, 38), 104: (22, 38), 105: (26, 38),
    106: (30, 38), 107: (33, 38), 108: (37, 38), 109: (41, 38), 110: (45, 38), 111: (50, 38), 112: (54, 38),
    113: (58, 38), 114: (62, 38), 115: (66, 38), 116: (70, 38), 117: (74, 38), 118: (78, 38), 119: (81, 38),
    120: (84, 38), 121: (87, 38), 122: (6, 36), 123: (12, 41), 124: (18, 41), 125: (22, 41), 126: (26, 41),
    127: (30, 41), 128: (33, 41), 129: (35, 41), 130: (37, 41), 131: (41, 41), 132: (44, 41), 133: (48, 41),
    134: (51, 41), 135: (54, 41), 136: (58, 41), 137: (62, 41), 138: (66, 41), 139: (68, 41), 140: (75, 41),
    141: (78, 41), 142: (81, 41), 143: (84, 41), 144: (87, 41), 145: (14, 45), 146: (16, 49), 147: (18, 47),
    148: (22, 47), 149: (27, 47), 150: (31, 47), 151: (35, 47), 152: (39, 47), 153: (43, 47), 154: (47, 47),
    155: (51, 47), 156: (56, 47), 157: (60, 47), 158: (64, 47), 159: (68, 47), 160: (72, 47), 161: (75, 47),
    162: (78, 47), 163: (81, 47), 164: (84, 47), 165: (87, 47), 166: (18, 53), 167: (22, 53), 168: (27, 53),
    169: (31, 53), 170: (35, 53), 171: (39, 53), 172: (43, 53), 173: (47, 53), 174: (51, 53), 175: (56, 53),
    176: (60, 53), 177: (64, 53), 178: (68, 53), 179: (72, 53), 180: (75, 53), 181: (78, 53), 182: (81, 53),
    183: (84, 53), 184: (87, 53), 185: (92, 53), 186: (20, 58), 187: (25, 58), 188: (31, 58), 189: (35, 58),
    190: (39, 58), 191: (43, 58), 192: (47, 58), 193: (51, 58), 194: (56, 58), 195: (60, 58), 196: (64, 58),
    197: (68, 58), 198: (74, 21), 199: (72, 41)
}



# FOR ORIGINAL LAYOUT
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20),
    (20, 21), (21, 22), (22, 23), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31),
    (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (31, 38), (38, 39), (39, 40),
    (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51),
    (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 60), (60, 61),
    (63, 64), (64, 65), (65, 66), (66, 67), (67, 68), (68, 69), (69, 70), (70, 71),
    (71, 72), (72, 73), (73, 74), (74, 75), (75, 76), (76, 77), (77, 78), (78, 79), (79, 80), (80, 81),
    (83, 84), (85, 86), (86, 87), (87, 88), (88, 89), (89, 90), (90, 91), (91, 92),
    (92, 93), (93, 94), (94, 95), (95, 96), (96, 97), (97, 98), (98, 99), (99, 100), (100, 101),
    (103, 104), (104, 105), (105, 106), (106, 107), (107, 108), (108, 109), (109, 110), (110, 111),
    (111, 112), (112, 113), (113, 114), (114, 115), (115, 116), (116, 117), (117, 118), (118, 119), (119, 120),
    (120, 121), (123, 124), (124, 125), (125, 126), (126, 127), (127, 128), (128, 129),
    (129, 130), (130, 131), (131, 132), (132, 133), (133, 134), (134, 135), (135, 136), (136, 137), (137, 138),
    (138, 139), (139, 140), (140, 141), (141, 142), (142, 143), (143, 144), (166, 147),
    (148, 149), (149, 150), (150, 151), (151, 152), (152, 153), (153, 154), (154, 155), (155, 156),
    (156, 157), (157, 158), (158, 159), (159, 160), (160, 161), (161, 162), (162, 163), (163, 164), (164, 165),
    (166, 167), (167, 168), (168, 169), (169, 170), (170, 171), (171, 172), (172, 173), (173, 174), (174, 175),
    (175, 176), (176, 177), (177, 178), (178, 179), (179, 180), (180, 181), (181, 182), (182, 183), (183, 184),
    (184, 185), (186, 187), (187, 188), (188, 189), (189, 190), (190, 191), (191, 192), (192, 193),
    (193, 194), (194, 195), (195, 196), (196, 197),
    (4, 24), (24, 42), (42, 62), (62, 82), (82, 102), (102, 123), (123, 146), (146, 186),
    (6, 25), (25, 43), (43, 63), (63, 83), (83, 103), (103, 124), (124, 147),
    (9, 11), (11, 28), (28, 46), (46, 66), (66, 86), (86, 106), (106, 127),
    (14, 31), (31, 49), (49, 69), (69, 89), (89, 109), (109, 131),
    (10, 17), (17, 34), (34, 52), (52, 72), (72, 92), (92, 112), (112, 135),
    (20, 37), (37, 55), (55, 75), (75, 95), (95, 115), (115, 138),
    (58, 78), (78, 98), (98, 118), (141, 162), (162, 181),
    (61, 81), (81, 101), (101, 121), (121, 144), (144, 165), (165, 184),
    (139, 159), (159, 178), (178, 197),
    (134, 155), (155, 174), (174, 193),
    (129, 151), (151, 170), (170, 189),
    (41, 59), (41, 60), (102, 122),
    (23, 40), (40, 57),
    (123, 145), (145, 146), (84, 85), (148, 147), (9, 11), (139, 199), (51, 198)
]

# # FOR ONEWAY SETUP
# edges = [
#      (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20),
#     (20, 21), (21, 22), (22, 23), (26, 25), (27, 26), (28, 27), (29, 28), (30, 29), (31, 30),
# (32, 31), (33, 32), (34, 33), (35, 34), (36, 35), (37, 36), (38, 37), 
# (38, 31), (39, 38), (40, 39), (148, 147), (56, 198), (139, 199), (122, 102),  (124, 123), (199, 140), (185, 184),


#     (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51),
#     (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (57, 58), (58, 59), (59, 60), (60, 61),
#     (64, 63), (65, 64), (66, 65), (67, 66), (68, 67), (69, 68), (70, 69), (71, 70),


# (72, 71), (73, 72), (74, 73), (75, 74), (76, 75), (77, 76), (78, 77), (79, 78), (80, 79), (81, 80),
#     (83, 84), (85, 86), (86, 87), (87, 88), (88, 89), (89, 90), (90, 91), (91, 92),
#     (92, 93), (93, 94), (94, 95), (95, 96), (96, 97), (97, 98), (98, 99), (99, 100), (100, 101),
#     (104, 103), (105, 104), (106, 105), (107, 106), (108, 107), (109, 108), (110, 109), (111, 110), (84, 85),  (102, 122),


# (112, 111), (113, 112), (114, 113), (115, 114), (116, 115), (117, 116), (118, 117), (119, 118), (120, 119),
# (121, 120), (123, 124), (124, 125), (125, 126), (126, 127), (127, 128), (128, 129),
#     (129, 130), (130, 131), (131, 132), (132, 133), (133, 134), (134, 135), (135, 136), (136, 137), (137, 138),
#     (138, 139), (140, 141), (141, 142), (142, 143), (143, 144),


#     (166, 167), (167, 168), (168, 169), (169, 170), (170, 171), (171, 172), (172, 173), (173, 174), (174, 175),
#     (175, 176), (176, 177), (177, 178), (178, 179), (179, 180), (180, 181), (181, 182), (182, 183), (183, 184),
#     (184, 185), (187, 186), (188, 187), (189, 188), (190, 189), (191, 190), (192, 191), (193, 192),
# (194, 193), (195, 194), (196, 195), (197, 196), (198, 57),

# (149, 148), (150, 149), (151, 150), (152, 151), (153, 152), (154, 153), (155, 154), 
# (156, 155), (157, 156), (158, 157), (159, 158), (160, 159), (161, 160), (162, 161), (163, 162), (164, 163), (165, 164),
    
    
#     # verticals
#     (4, 24), (24, 42), (42, 62), (62, 82), (82, 102), (102, 123), (146, 166), (166, 186),
#     (147, 166),  (6, 25), (25, 43), (43, 63), (63, 83), (83, 103), (103, 124), (124, 147),
#     (11, 9), (11, 28), (28, 46), (46, 66), (66, 86), (86, 106), (106, 127),
#     (14, 31), (31, 49), (49, 69), (69, 89), (89, 109), (109, 131),
#     (10, 17), (17, 34), (34, 52), (52, 72), (72, 92), (92, 112), (112, 135),  

#     (20, 37), (37, 55), (55, 75), (75, 95), (95, 115), (115, 138),
#     (58, 78), (78, 98), (98, 118), (141, 162), (162, 181),
#     (61, 81), (81, 101), (101, 121), (121, 144), (144, 165), (165, 184),
#     (139, 159), (159, 178), (178, 197), (0, 1), (1, 2), (2, 3), (3, 4),


#     (134, 155), (155, 174), (174, 193),
#     (129, 151), (151, 170), (170, 189),
#     (41, 59), (41, 60),
#     (23, 40), (40, 57),
#     (123, 145), (145, 146),   (9, 11),  
    


#     (24, 4), (42, 24), (62, 42), (82, 62), (102, 82), (123, 102),
#     (4, 3), (3, 2), (2, 1), (1, 0),
#     (166, 147), (147, 124), (124, 103), (103, 83), (83, 63), (63, 43), (43, 25), (25, 6),
#     (127, 106), (106, 86), (86, 66), (66, 46), (46, 28), (28, 9), 


#     (189, 170), (170, 151), (151, 129),
#     (193, 174), (174, 155), (155, 134),
#     (131, 109), (109, 89), (89, 69), (69, 49), (49, 31), (31, 14), (11, 9),
#     (135, 112), (112, 92), (92, 72), (72, 52), (52, 34), (34, 17), (17, 10), (197, 178), (178, 159), (159, 139), (138, 115), (115, 95), (95, 75), (75, 55), (55, 37), (37, 20),
    
#     (181, 162), (162, 141), (118, 98), (98, 78), (78, 58), (57, 40), (40, 23), (59, 41), (60, 41), (184, 165), (165, 144), (144, 121), (121, 101), (101, 81), (81, 61), (9, 11),
#     (186, 166), (166, 146), (146, 145), (145, 123), (147, 166)

# ]


G = create_store_network(pos, edges)
# print(G)
shortest_path_dict = get_all_shortest_path_dicts(G)
mu = 0.07
sigma = 0.76
entrance_nodes = [0, 11, 122, 10, 41, 185]
exit_nodes = [0, 11, 122, 10, 41, 185]
intersections = [9, 57, 189, 170, 151, 129, 106, 86, 66, 46, 28, 127, 193, 174, 155, 134, 112, 92, 72, 52, 34, 17, 197, 178, 159, 139, 115, 95, 75, 55, 37, 23, 49, 58, 78, 98, 118, 121, 101, 81, 61, 144, 165, 184, 181, 162, 141, 131, 109, 89, 69, 31, 14, 20, 40, 135, 138, 124, 147]
# item_nodes = [node for node in range(199) if node not in entrance_nodes and node not in exit_nodes]
item_nodes = [node for node in range(200) if node not in entrance_nodes and intersections]



def paths_generator_from_actual_paths(all_paths):
    num_paths = len(all_paths)
    while True:
        i = np.random.randint(0, num_paths)
        yield all_paths[i]


def path_generator_from_transition_matrix(tmatrix: List[List[int]], shortest_path_dict: Dict):
    while True:
        yield zone_path_to_full_path(create_one_path(tmatrix), shortest_path_dict)


def get_transition_matrix(all_paths, num_states):
    n = num_states + 1  # number of states

    transition_matrix = [[0] * n for _ in range(n)]
    for path in all_paths:
        for (i, j) in zip(path, path[1:]):
            transition_matrix[i][j] += 1
        transition_matrix[path[-1]][n - 1] += 1  # ending

    # now convert to probabilities:
    for row in transition_matrix:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]
    return transition_matrix


def zone_path_to_full_path(zone_path, shortest_path_dict):
    full_path_dict = []
    for start, end in zip(zone_path[:-1], zone_path[1:]):
        full_path_dict += shortest_path_dict[start][end]
    return full_path_dict


def zone_path_to_full_path_multiple_paths(zone_path, shortest_path_dict):
    """Use this if there are shortest_path_dict gives multiple shortest_paths for each source and target"""
    L = []
    for start, end in zip(zone_path[:-1], zone_path[1:]):
        shortest_paths = shortest_path_dict[start][end]
        num_shortest_paths = len(shortest_paths)
        if num_shortest_paths > 0:
            i = np.random.randint(num_shortest_paths)
        else:
            i = 0
        L += shortest_paths[i]
    return L

def sample_num_products_in_basket_batch(mu, sigma, num_baskets):
    """
    Sample the number of items in basket using Mixed Poisson Log-Normal distribution.
    Reference: Sorensen H, Bogomolova S, Anderson K, Trinh G, Sharp A, Kennedy R, et al.
    Fundamental patterns of in-store shopper behavior. Journal of Retailing and Consumer Services. 2017;37:182–194.

    :param mu: Mean value of the underlying normal distribution
    :param sigma: Standard deviation of the underlying normal distribution
    :param num_baskets: number of baskets to sample
    :return: List of length num_baskets with the number of items in each basket
    """

    norm = np.random.lognormal(mean=mu, sigma=sigma, size=3 * num_baskets)
    num_items = np.random.poisson(norm)
    num_items = num_items[num_items > 0]
    assert len(num_items) >= num_baskets, \
        f"Somehow didn't get the enough non-zero baskets ({num_items} <= {num_baskets} (size))"
    return num_items[:num_baskets]


def create_random_item_paths(num_items, entrance_nodes, exit_nodes, item_nodes):
    """
    Create random item path based on the number of items in each basket and the shelves were items are located.
    We choose items uniformly at random from all item_nodes.
    We also choose a random entrance node, till node, and exit node (sampled uniformly at random from the
    corresponding nodes).
    """
    num_baskets = len(num_items)
    random_entrance_nodes = np.random.choice(entrance_nodes, size=num_baskets)
    # random_till_nodes = np.random.choice(till_nodes, size=num_baskets)
    random_exit_nodes = np.random.choice(exit_nodes, size=num_baskets)
    concatenated_baskets = np.random.choice(item_nodes, size=np.sum(num_items))
    break_points = np.cumsum(num_items)
    item_paths = []
    start = 0
    i = 0
    for end in break_points:
        entrance = random_entrance_nodes[i]
        # till = random_till_nodes[i]
        exit = random_exit_nodes[i]
        basket = [entrance] + list(concatenated_baskets[start:end]) + [exit]
        item_paths.append(basket)
        start = end
        i += 1
    return item_paths


def sythetic_paths_generator(mu, sigma, entrance_nodes, exit_nodes, item_nodes,
                                                shortest_path_dict, batch_size=1000000):
    """The synthetic path generator generates a random customer path as follows:
    First, it samples the size K of the shopping basket using a log-normal random variable with parameter mu and sigma.
    Second, it chooses a random entrance node as the first node v_1 in the path.
    Third, it samples K random item nodes, chosen uniformly at random with replacement from item_nodes, which we denote by
    v_2, ... v_K+1.
    Fourth, it samples a random till node and exit node, which we denote by v_K+2 and v_K+3.
    The sequence v_1, ..., v_K+3 is a node sequence where the customer bought items, along the the entrance, till and exit
    nodes that they visited.
    Finally, we convert this sequence to a full path on the network using the shortest paths between consecutive nodes
    in the sequence.
    We use shortest_path_dict for this.
    For more information, see the Data section in https://arxiv.org/pdf/2010.07868.pdf

    The batch_size specifies how many paths we generate in each batch (for efficiency reasons).
    """
    mylist = []
    while True:
        num_items = sample_num_products_in_basket_batch(mu, sigma, batch_size)
        item_paths = create_random_item_paths(num_items, entrance_nodes, exit_nodes, item_nodes)
        for item_path in item_paths:
            full_path = zone_path_to_full_path_multiple_paths(item_path, shortest_path_dict)
            mylist.append(full_path)
        break
    return mylist


def get_next_term(num_states, trow):
    return random.choices(range(num_states), trow)[0]


def create_one_path(tmatrix: List[List[int]]):
    num_states = len(tmatrix)
    start_term = 0
    end = num_states - 1
    chain = [start_term]
    length = 1
    while True:
        current_position = get_next_term(num_states, tmatrix[chain[-1]])
        if current_position == end:
            break
        elif length > 100000:
            print('Generated is over 100000 stops long. Something must have gone wrong!')
            break
        chain.append(current_position)
        length += 1
    return chain


def replace_till_zone(path, till_zone, all_till_zones):
    assert path[-1] == till_zone, f'Final zone is not {till_zone}, but {path[-1]}'
    path[-1] = np.random.choice(all_till_zones)
    return path


def get_path_generator(path_generation: str = 'empirical', G: Optional[nx.Graph]=None,
                       full_paths: Optional[List[List[int]]]=None,
                       zone_paths: Optional[List[List[int]]]=None,
                       synthetic_path_generator_args: Optional[list] = None):
    """Create path generator functions.
    Note that a zone path is a sequence of zones that a customer purchased items from, so consecutive zones in the sequence
    may not be adjacent in the store graph. We map the zone path to the full shopping path by assuming that
    customers walk shortest paths between purchases."""

    # Decide how paths are generated
    if path_generation == 'empirical':
        path_generator_function = paths_generator_from_actual_paths
        if full_paths is not None:
            path_generator_args = [full_paths]
        else:
            assert zone_paths is not None, "If you use path_generation='empirical', you need to specify either zone_paths or full_paths"
            assert G is not None, "If you use path_generation='empirical' with zone_paths, you need to input the store network G"
            shortest_path_dict = dict(nx.all_pairs_dijkstra_path(G))
            shopping_paths = [zone_path_to_full_path(path, shortest_path_dict) for path in zone_paths]
            full_paths = [zone_path_to_full_path(path, shortest_path_dict) for path in shopping_paths]
            path_generator_args = [full_paths]
    elif path_generation == 'synthetic':
        assert synthetic_path_generator_args is not None, \
            "If you use path_generation='synthetic', " \
            "you need to input synthetic_path_generator_args=" \
            "[mu, sigma, entrance_nodes, till_nodes, exit_nodes, item_nodes, shortest_path_dict]"
        assert type(synthetic_path_generator_args) is list, \
            "If you use path_generation='synthetic', " \
            "you need to input synthetic_path_generator_args=" \
            "[mu, sigma, entrance_nodes, till_nodes, exit_nodes, item_nodes, shortest_path_dict]"
        assert len(synthetic_path_generator_args) == 7, \
            "If you use path_generation='synthetic', " \
            "you need to input synthetic_path_generator_args=" \
            "[mu, sigma, entrance_nodes, till_nodes, exit_nodes, item_nodes, shortest_path_dict]"
        path_generator_function = sythetic_paths_generator
        path_generator_args = synthetic_path_generator_args  # [mu, sigma, entrance_nodes,
        # till_nodes, exit_nodes, item_nodes, shortest_path_dict]
    elif path_generation == 'tmatrix':
        assert zone_paths is not None, "If you use path_generation='tmatrix', you need to input zone_paths"
        assert G is not None, "If you use path_generation='tmatrix', you need to input the store network G"
        shortest_path_dict = dict(nx.all_pairs_dijkstra_path(G))
        shopping_paths = [zone_path_to_full_path(path, shortest_path_dict) for path in zone_paths]
        tmatrix = get_transition_matrix(shopping_paths, len(G))
        path_generator_function = path_generator_from_transition_matrix
        path_generator_args = [tmatrix, shortest_path_dict]
    else:
        raise ValueError(f'Unknown path_generation scheme == {path_generation}')
    return path_generator_function, path_generator_args


zone_paths = sythetic_paths_generator(mu, sigma, entrance_nodes, exit_nodes, item_nodes, shortest_path_dict)

# file_path = '10^6_DIRECTIONAL.json' # for directed
file_path = '10_6.json' 

with open(file_path, 'w') as json_file:
    json.dump(zone_paths, json_file)

print(f"The list has been written to {file_path}")