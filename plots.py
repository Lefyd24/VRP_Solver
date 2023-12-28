import itertools
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

def plot_vehicles(vehicles: list, nodes: list, distance_matrix, title):
    color_palette = itertools.cycle(sns.color_palette("tab20", n_colors=len(vehicles)))
    depot_color = 'red'  # Unique color for the depot

    G = nx.DiGraph()

    node_colors = {}
    edge_colors = []
    for vehicle in vehicles[:10]:
        route = vehicle.route
        route_color = next(color_palette)
        for i, node in enumerate(route):
            node_id = node.ID
            if node_id not in node_colors:
                node_colors[node_id] = depot_color if node_id == 0 else route_color
            if i < len(route) - 1:
                edge_colors.append(route_color)
                G.add_edge(node.ID, route[i+1].ID, weight=distance_matrix[node.ID, route[i+1].ID])

    # Scale up coordinates
    scale_factor = 5  # Adjust this factor as needed
    pos = {int(customer.ID): (customer.x * scale_factor, customer.y * scale_factor) for customer in nodes}

    plt.figure(figsize=(20, 12))
    nx.draw_networkx_nodes(G, pos, node_size=450, node_color=[node_colors.get(node) for node in G.nodes()])
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color=edge_colors, width=1)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

    plt.title(f"First 10 Vehicle Routes - Solution: {title}", fontsize=25, fontweight='bold', pad=30)
    plt.xlabel("X Coordinate", fontsize=17, labelpad=20)
    plt.ylabel("Y Coordinate", fontsize=17, labelpad=20)

    plt.savefig(f'plots/{title}.png')