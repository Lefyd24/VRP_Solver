import pandas as pd
import numpy as np
from plots import plot_vehicles
from helpers import *
from two_opt import two_opt_local_search
from copy import copy, deepcopy
import random
import os
import datetime as dt
from colorama import Fore, Style 
import time

# Classes
class Customer:
    """Represents a customer with a demand and location."""

    def __init__(self, customer_id, x, y, demand):
        self.ID = int(customer_id)
        self.x = x
        self.y = y
        self.demand = demand

    def __repr__(self):
        return f"Customer(ID: {self.ID}, Demand: {self.demand}, Coordinates: ({self.x}, {self.y}))"
    
    def __str__(self):
        return f"Customer {self.ID}"
    
class Vehicle:
    """Represents a vehicle with capacity and current load."""
    _id_counter = 0

    def __init__(self, capacity, empty_weight, distance_matrix):
        # make the id auto-incrementing
        self.ID =  Vehicle._id_counter
        Vehicle._id_counter += 1
        self.capacity = capacity
        self.empty_weight = empty_weight
        self.route = []
        self.load = 0
        self.total_distance = 0
        self.cost = 0
        self.distance_matrix = distance_matrix
        self.sliding_demand = dict()

    def __repr__(self):
            return f"Vehicle(Capacity: {self.capacity}, Load: {self.load})"
    
    def add_customer(self, customer):
        """Adds a customer to the vehicle's route if capacity allows."""
        if self.load + customer.demand <= self.capacity:
            self.route.append(customer)
            self.load += customer.demand
            self.sliding_demand[customer.ID] = self.load
            if len(self.route) > 1:
                new_distance = self.distance_matrix[self.route[-1].ID][self.route[-2].ID]
                self.total_distance += new_distance
                self.update_cost()
            else: # calculate distance from warehouse (0,0) to first customer
                self.total_distance += self.distance_matrix[self.route[-1].ID][0]
                self.update_cost()
            return True
        return False
    
    def update_cost(self, update_object=True):
            """
            Updates the cost of the route based on the current state of the route and the distance matrix.

            Parameters:
            - update_object (bool): Flag indicating whether to update the object attributes (load, total_distance, cost).

            Returns:
            - tn_km (float): The updated cost of the route.
            """
            tot_dem = sum(n.demand for n in self.route)
            tot_load = self.empty_weight + tot_dem
            tn_km = 0
            km = 0
            for i in range(len(self.route) - 1):
                from_node = self.route[i]
                to_node = self.route[i+1]
                km += self.distance_matrix[from_node.ID][to_node.ID]
                tn_km += self.distance_matrix[from_node.ID][to_node.ID] * tot_load
                tot_load -= to_node.demand
                # Update the sliding demand with the total load until this node
                self.sliding_demand[to_node.ID] = tot_load
            # Remove nodes that have been removed from the route (in cases of local search moves)
            for node in list(self.sliding_demand.keys()):
                if node not in [customer.ID for customer in self.route]:
                    self.sliding_demand.pop(node)
            if update_object:
                self.load = tot_dem
                self.total_distance = km
                self.cost = tn_km
            return tn_km

    def recalculate_route(self):
        """Recalculate the total distance and load of the vehicle after a route change."""
        self.total_distance = 0
        self.load = sum(customer.demand for customer in self.route)
        for i in range(len(self.route) - 1):
            self.total_distance += euclidean_distance(self.route[i].x, self.route[i].y, self.route[i+1].x, self.route[i+1].y)
        #self.update_cost()

    def find_best_swap(self, other_vehicle, matrix):
        best_swap = None
        for first_route_node_index in range(1, len(self.route)-1):

            start_of_second_index = first_route_node_index + 1 if self.ID == other_vehicle.ID else 1

            for second_route_node_index in range(start_of_second_index, len(other_vehicle.route)-1):
                
                a1 = self.route[first_route_node_index - 1]
                b1 = self.route[first_route_node_index]
                c1 = self.route[first_route_node_index + 1]
                a2 = other_vehicle.route[second_route_node_index - 1]
                b2 = other_vehicle.route[second_route_node_index]
                c2 = other_vehicle.route[second_route_node_index + 1]
                move_cost = None
                cost_change_first_route = None
                cost_change_second_route = None
                # Same route swap
                if self.ID == other_vehicle.ID:
                    if first_route_node_index == second_route_node_index - 1:
                        # Nodes are neighbors
                        cost_removed = matrix[a1.ID][b1.ID]*self.sliding_demand[b1.ID] + matrix[b1.ID][b2.ID]*other_vehicle.sliding_demand[b2.ID] + matrix[b2.ID][c2.ID]*other_vehicle.sliding_demand[c2.ID]
                        cost_added = matrix[a1.ID][b2.ID]*other_vehicle.sliding_demand[b2.ID] + matrix[b2.ID][b1.ID]*self.sliding_demand[b1.ID] + matrix[b1.ID][c2.ID]*other_vehicle.sliding_demand[c2.ID]
                        move_cost = cost_added - cost_removed
                    else:
                        # Nodes are not neighbors
                        cost_removed_1 = matrix[a1.ID][b1.ID]*self.sliding_demand[b1.ID] + matrix[b1.ID][c1.ID]*self.sliding_demand[c1.ID]
                        cost_added_1 = matrix[a1.ID][b2.ID]*other_vehicle.sliding_demand[b2.ID] + matrix[b2.ID][c1.ID]*self.sliding_demand[c1.ID]
                        cost_removed_2 = matrix[a2.ID][b2.ID]*other_vehicle.sliding_demand[b2.ID] + matrix[b2.ID][c2.ID]*other_vehicle.sliding_demand[c2.ID]
                        cost_added_2 = matrix[a2.ID][b1.ID]*self.sliding_demand[b1.ID] + matrix[b1.ID][c2.ID]*other_vehicle.sliding_demand[c2.ID]
                        move_cost = cost_added_1 + cost_added_2 - cost_removed_1 - cost_removed_2
                else:
                    if (self.load - b1.demand + b2.demand > self.capacity) or (other_vehicle.load - b2.demand + b1.demand > other_vehicle.capacity):
                        continue
                    else:
                        cost_removed_a = matrix[a1.ID][b1.ID]*self.sliding_demand[b1.ID] + matrix[b1.ID][c1.ID]*self.sliding_demand[c1.ID]
                        cost_added_a = matrix[a1.ID][b2.ID]*other_vehicle.sliding_demand[b2.ID] + matrix[b2.ID][c1.ID]*self.sliding_demand[c1.ID]
                        cost_removed_b = matrix[a2.ID][b2.ID]*other_vehicle.sliding_demand[b2.ID] + matrix[b2.ID][c2.ID]*other_vehicle.sliding_demand[c2.ID]
                        cost_added_b = matrix[a2.ID][b1.ID]*self.sliding_demand[b1.ID] + matrix[b1.ID][c2.ID]*other_vehicle.sliding_demand[c2.ID]
                        move_cost = cost_added_a + cost_added_b - cost_removed_a - cost_removed_b
                        cost_change_first_route = cost_added_a - cost_removed_a
                        cost_change_second_route = cost_added_b - cost_removed_b
                        
                if move_cost <= 0.0001:
                    if best_swap is None:
                        best_swap = (first_route_node_index, second_route_node_index, move_cost, cost_change_first_route, cost_change_second_route)
                    elif move_cost < best_swap[2]:
                        best_swap = (first_route_node_index, second_route_node_index, move_cost, cost_change_first_route, cost_change_second_route)
        if best_swap is not None:
            #print(f"Swapping: {best_swap} | Vehicles: {self.ID}, {other_vehicle.ID}")
            self.inter_route_swap(other_vehicle, best_swap[0], best_swap[1])
            #return best_swap[2], best_swap[3], best_swap[4]
    
    def perform_swap(self, other_vehicle):
        no_swaps = 0
        for idx_i, node_i in enumerate(self.route):
            for idx_j, node_j in enumerate(other_vehicle.route):
                if (self.ID == other_vehicle.ID and node_i == node_j) or (node_i.ID == 0 or node_j.ID == 0):
                    continue
                if self.load - node_i.demand + node_j.demand <= self.capacity and other_vehicle.load - node_j.demand + node_i.demand <= other_vehicle.capacity:
                    # check if the overall cost is reduced (cost of self minus (-) cost of other vehicle)
                    current_vehicle_copy = deepcopy(self)
                    other_vehicle_copy = deepcopy(other_vehicle)
                    # cost before swap
                    current_vehicle_copy_cost_before = current_vehicle_copy.cost
                    other_vehicle_copy_cost_before = other_vehicle_copy.cost
                    # swap
                    current_vehicle_copy.route[idx_i], other_vehicle_copy.route[idx_j] = other_vehicle_copy.route[idx_j], current_vehicle_copy.route[idx_i]
                    # cost after swap
                    current_vehicle_copy.update_cost()
                    current_vehicle_copy_cost_after = current_vehicle_copy.cost
                    other_vehicle_copy.update_cost()
                    other_vehicle_copy_cost_after = other_vehicle_copy.cost
                    # calculate overall cost change
                    cost_change = current_vehicle_copy_cost_after - current_vehicle_copy_cost_before + other_vehicle_copy_cost_after - other_vehicle_copy_cost_before

                    if cost_change <= 0.00001 and current_vehicle_copy.load <= current_vehicle_copy.capacity and other_vehicle_copy.load <= other_vehicle_copy.capacity:
                        # Apply the swap
                        #print(f"Swapping: {node_i}, {node_j} | Vehicles: {self.ID}, {other_vehicle.ID}")
                        self.route[idx_i], other_vehicle.route[idx_j] = other_vehicle.route[idx_j], self.route[idx_i]
                        self.update_cost()
                        other_vehicle.update_cost()
                        no_swaps += 1
                else:
                    continue
        return no_swaps
                        
    def inter_route_swap(self, other_vehicle, i, k):
        """
        Swaps two nodes between two routes.

        Args:
            other_vehicle (Vehicle): The other vehicle.
            i (int): The first node's index.
            k (int): The second node's index.
        """
        first_node = self.route[i]
        second_node = other_vehicle.route[k]
        old_sliding = self.sliding_demand.copy()
        print(f"Swapping: {i} ({first_node}), {k} ({second_node}) | Vehicles: {self.ID}, {other_vehicle.ID}")

        #print(f"{Fore.YELLOW}Previous cost: ", self.cost, other_vehicle.cost)
        #print(f"Sliding demand: {self.sliding_demand.items()}\n\n{other_vehicle.sliding_demand}{Style.RESET_ALL}")
        
        self.route[i] = second_node
        other_vehicle.route[k] = first_node
        
        if self.ID != other_vehicle.ID:
            self.update_cost()
            other_vehicle.update_cost()
            #print(f"{Fore.GREEN}New cost: {self.cost} , {other_vehicle.cost}")
            #print(f"Sliding demand: {self.sliding_demand}\n\n{other_vehicle.sliding_demand}{Style.RESET_ALL}")
            #print(old_sliding == self.sliding_demand)
        else:
            self.update_cost()
        
    def find_best_relocation(self, other_vehicle, matrix):
        """
        Finds the best relocation between two routes.

        Args:
            other_vehicle (Vehicle): The other vehicle.

        Returns:
            tuple: The best relocation.
        """
        #print(f"Vehicle {self.id} and Vehicle {other_vehicle.id}")
        best_relocation = None
        no_relocations = 0
        for first_route_node_index in range(len(self.route)-1):
            for second_route_node_index in range(len(other_vehicle.route)-1):

                if (first_route_node_index == 0 or second_route_node_index == 0) or \
                    (self.ID == other_vehicle.ID and (second_route_node_index == first_route_node_index) or (second_route_node_index == first_route_node_index - 1) or (second_route_node_index == first_route_node_index + 1)):
                    continue

                a1 = self.route[first_route_node_index - 1]
                a2 = self.route[first_route_node_index]
                a3 = self.route[first_route_node_index + 1]

                b1 = other_vehicle.route[second_route_node_index]
                b3 = other_vehicle.route[second_route_node_index + 1]

                move_cost = None
                cost_change_first_route = None
                cost_change_second_route = None

                if self.ID != other_vehicle.ID:
                    if other_vehicle.load + a2.demand > other_vehicle.capacity:
                        continue
                else:
                    cost_added = matrix[a1.ID][a3.ID]*self.sliding_demand[a3.ID] + matrix[b1.ID][a2.ID]*self.sliding_demand[a2.ID] + matrix[a2.ID][b3.ID]*other_vehicle.sliding_demand[b3.ID]
                    cost_removed = matrix[a1.ID][a2.ID]*self.sliding_demand[a2.ID] + matrix[a2.ID][a3.ID]*self.sliding_demand[a3.ID] + matrix[b1.ID][b3.ID]*other_vehicle.sliding_demand[b3.ID]

                    move_cost = cost_added - cost_removed

                    cost_change_first_route = matrix[a1.ID][a3.ID]*self.sliding_demand[a3.ID] - matrix[a1.ID][a2.ID]*self.sliding_demand[a2.ID] - matrix[a2.ID][a3.ID]*self.sliding_demand[a3.ID]
                    cost_change_second_route = matrix[b1.ID][a2.ID]*self.sliding_demand[a2.ID] + matrix[a2.ID][b3.ID]*other_vehicle.sliding_demand[b3.ID] - matrix[b1.ID][b3.ID]*other_vehicle.sliding_demand[b3.ID]

                    if move_cost < 0:
                        if best_relocation is None:
                            best_relocation = (first_route_node_index, second_route_node_index, move_cost, cost_change_first_route, cost_change_second_route)
                        elif move_cost < best_relocation[2]:
                            best_relocation = (first_route_node_index, second_route_node_index, move_cost, cost_change_first_route, cost_change_second_route)
        if best_relocation is not None:
            #print(f"Relocating: {best_relocation} | Vehicles: {self.ID}, {other_vehicle.ID}")
            self.inter_route_relocate(other_vehicle, best_relocation)
            no_relocations += 1
        return no_relocations

    def inter_route_relocate(self, other_vehicle, best_relocation):
        """
        Relocates a node from one route to another.

        Args:
            other_vehicle (Vehicle): The other vehicle.
            i (int): The index of the node of first route to relocate from.
            k (int): The index of the node of second route to relocate to.
        """
        i = best_relocation[0]
        k = best_relocation[1]
        
        if self.ID != other_vehicle.ID:
            other_vehicle.route.insert(k+1, self.route[i])
            self.route.pop(i)
            self.update_cost()
            other_vehicle.update_cost()
        else:
            new_node = self.route[i]
            self.route.pop(i)
            if i < k:
                other_vehicle.route.insert(k, new_node)
            else:
                other_vehicle.route.insert(k+1, new_node)
            self.update_cost()


with open('./dataset/Instance.txt', 'r') as ins:
    data = ins.readlines()

data = [line.strip().split(',') for line in data]
df = pd.DataFrame(data[5:], columns=data[4])
df = df.apply(pd.to_numeric, errors='coerce')

# Create nodes
nodes = [Customer(row['ID'], row['XCOORD'], row['YCOORD'], row['DEMAND']) for _, row in df.iterrows()]

# Distance Matrix
distance_matrix = create_node_matrix(nodes, type_matrix="distance")

# Initialize the first solution via the Nearest Neighbor heuristic. 
# For populating the vehicles, the First-Fit Bin Packing technique is used.
ff_dist_vehicles = first_fit_nn(nodes, distance_matrix, Vehicle)

def VND(vehicles):
    """
    Variable Neighborhood Descent (VND) algorithm for solving our vehicle routing problem.

    Parameters:
    - vehicles (list): A list of Vehicle objects representing the initial solution.

    Returns:
    - best_solution (list): A list of Vehicle objects representing the best solution found.
    - best_cost (float): The cost of the best solution found.
    - VND_iterator (int): The number of iterations performed by the VND algorithm.
    - search_trajectory (list): A list of tuples representing the search trajectory. Each tuple contains the move type (0, 1, or 2) and the corresponding cost.

    Move Types:
    - 0: 2-opt
    - 1: swap
    - 2: node relocation
    """
    vehicles_copy = deepcopy(vehicles)
    VND_iterator = 0
    kmax = 3
    k = 0
    search_trajectory = []
    best_cost = sum(v.cost for v in vehicles_copy)
    best_solution = deepcopy(vehicles_copy)
    
    while k <= kmax:
        if k == 2 or k == 3:
            total_relocations = 0
            for _ in range(1, 18): # We apply the inter-relocation optimization 17 times (empirical optimal)
                for vehicle in vehicles_copy:
                    for other_vehicle in vehicles_copy:
                        no_relocations = vehicle.find_best_relocation(other_vehicle, distance_matrix)
                        total_relocations += no_relocations
            print(f"Total relocations: {total_relocations}")
            VND_iterator += 1
            cost = sum(v.cost for v in vehicles_copy)
            search_trajectory.append((k, cost))
            
            if cost < best_cost:
                best_cost = cost
                best_solution = deepcopy(vehicles_copy)
                k = 0
            else:
                k += 1

        elif k == 1:
            total_swaps = 0
            for vehicle in vehicles_copy:
                for other_vehicle in vehicles_copy:
                    no_swaps = vehicle.perform_swap(other_vehicle)
                    total_swaps += no_swaps
            print(f"Total swaps: {total_swaps}")
            VND_iterator += 1
            cost = sum(v.cost for v in vehicles_copy)
            search_trajectory.append((k, cost))
            
            if cost < best_cost:
                best_cost = cost
                best_solution = deepcopy(vehicles_copy)
                k = 0
            else:
                k += 1
        elif k == 0:
            vehicles_copy = two_opt_local_search(vehicles_copy, num_iter=15000)
            VND_iterator += 1
            cost = sum(v.cost for v in vehicles_copy)
            search_trajectory.append((k, cost))
            
            if cost < best_cost:
                best_cost = cost
                best_solution = deepcopy(vehicles_copy)
                k = 0
            else:
                k += 1
    return best_solution, best_cost, VND_iterator, search_trajectory


if not os.path.exists('solutions'):
    os.makedirs('solutions')
if not os.path.exists('plots'):
    os.makedirs('plots') 

start = time.time()
random.seed(3)
best_solution, best_cost, VND_iterator, search_trajectory = VND(ff_dist_vehicles)
print(f"Best cost: {Fore.GREEN}{best_cost}{Style.RESET_ALL}")
print(f"Number of vehicles: {len(best_solution)}")
print(f"Number of VND iterations: {VND_iterator}")
print(f"Search trajectory: {search_trajectory}")
plot_vehicles(best_solution, nodes, distance_matrix, "VND")


end = time.time()
curr_datetime = dt.datetime.now().strftime("%d%m%Y%H%M")
# Store solution
with open(f'solutions/FF_OPT_VND_{curr_datetime}.txt', 'w') as f:
    f.write("Cost:\n")
    f.write(f"{best_cost}\n")
    f.write("Routes:\n")
    f.write(f"{len(best_solution)}\n")
    for vehicle in best_solution:
        route = ",".join(str(int(cust.ID)) for cust in vehicle.route)
        f.write(f"{route}\n")
print(f"Time elapsed: {end - start}s (or {(end - start)/60} minutes)")

plot_vehicles(best_solution, nodes, distance_matrix, "FF_OPT_VND_" + curr_datetime)