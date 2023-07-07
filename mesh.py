import numpy as np
from classes import Node, Q4


def mesh_sign(node_start_num, elem_start_num, start_position, length, width, t, m, n, bc, E1, v1, E2, v2, s_mode):
    nodes = {}
    for i in range(m + 1):
        for j in range(n + 1):
            node_num = node_start_num + i * (n + 1) + j
            position = (start_position[0] + (length / m) * i, start_position[1] - (width / n) * j)
            displacement, force = bc.get(position, ((np.nan, np.nan), (0, 0)))
            nodes[node_num] = Node(node_num, position, force, displacement)

    elements = {}
    for i in range(m):
        for j in range(n):
            num = elem_start_num + i * n + j
            n1 = node_start_num + (i + 1) * (n + 1) + j + 1
            n2 = node_start_num + (i + 1) * (n + 1) + j
            n3 = node_start_num + i * (n + 1) + j
            n4 = node_start_num + i * (n + 1) + j + 1
            e_nodes = (n1, n2, n3, n4)
            condition = 0.5 * (nodes[n1].position[1] + nodes[n2].position[1]) > start_position[1] - width / 2
            elements[num] = Q4(num, e_nodes, E1, v1, t, s_mode) if condition \
                else Q4(num, e_nodes, E2, v2, t, s_mode)

    return nodes, elements
