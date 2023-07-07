import copy
import matplotlib.pyplot as plt


def plot_output(nodes, elements, d, exaggeration=1):
    fig, ax = plt.subplots()

    new_nodes = copy.deepcopy(nodes)
    for n, node in new_nodes.items():
        new_position = (node.position[0] + exaggeration * d[int(2 * (n - 1))],
                        node.position[1] + exaggeration * d[int(2 * (n - 1) + 1)])
        node.position = new_position

    for element in elements.values():
        new_node_positions = [new_nodes[node].position for node in element.nodes]
        old_node_positions = [nodes[node].position for node in element.nodes]
        ax.plot(*zip(*old_node_positions, old_node_positions[0]), color='blue', zorder=1)
        ax.plot(*zip(*new_node_positions, new_node_positions[0]), color='black', zorder=1)

    ax.scatter(*zip(*[node.position for node in nodes.values()]), marker='o', linewidths=0.5, color='blue', zorder=2)
    ax.scatter(*zip(*[node.position for node in new_nodes.values()]), marker='o', linewidths=0.5, color='red', zorder=2)

    for i, node in enumerate(nodes.values()):
        ax.annotate(str(i + 1), node.position, textcoords="offset points", xytext=(-10, -10), ha='center')

    ax.set(xlabel="X", ylabel="Y", title='Elements after deformation', aspect=1,
           xlim=(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1), ylim=(ax.get_ylim()[0] - 0.5, ax.get_ylim()[1] + 0.5))

    plt.show()
