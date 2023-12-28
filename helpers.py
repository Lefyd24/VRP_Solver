import numpy as np
import itertools

def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def create_node_matrix(nodes, type_matrix="distance"):
    """
    Creates a distance matrix for the given nodes.
    """
    n_nodes = len(nodes)
    node_matrix = np.zeros((n_nodes, n_nodes))

    for (i, node1), (j, node2) in itertools.product(enumerate(nodes), repeat=2):
        if i != j:
            distance = euclidean_distance(node1.x, node1.y, node2.x, node2.y)
            #  The demand component calculation is taken from the destination node to showcase 
            #  the efficiency from the perspective of traveling from our current node to the destination node.
            if node2.demand == 0: # Avoid division by zero
                node_matrix[i, j] = float("inf") # we don't care about that distance when the vehicle is about to travel back to the warehouse
            else:
                if type_matrix == "distance":
                    node_matrix[i, j] = distance
                elif type_matrix == "ratio":
                    node_matrix[i, j] = distance / node2.demand
    return node_matrix

def first_fit_nn(nodes, matrix, Vehicle: object):
    unvisited_customers = nodes[1:].copy()
    warehouse = nodes[0]
    vehicles = []

    while unvisited_customers:
        vehicle = Vehicle(8, 6, matrix)
        route = [warehouse]  # Starting from the warehouse (index 0)
        vehicle.add_customer(warehouse) # Add warehouse to the route
        while unvisited_customers: # While there are unvisited customers
            # Find the nearest customer to the last customer in the route
            nearest_customer = min(unvisited_customers, key=lambda x: matrix[int(route[-1].ID), int(x.ID)]
                                                        if matrix[int(route[-1].ID), int(x.ID)] != 0
                                                        else np.inf)
            
            if vehicle.add_customer(nearest_customer): # If the customer can be added to the route
                unvisited_customers.remove(nearest_customer) # Remove the customer from the unvisited list
                route.append(nearest_customer) # Add the customer to the route
            else:
                break
        
        vehicles.append(vehicle) # Add the vehicle to the list of vehicles

    return vehicles



