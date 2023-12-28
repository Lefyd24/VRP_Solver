import numpy as np
import random
from copy import deepcopy
from colorama import Fore, Style

def two_opt_local_search(vehicles, num_iter=1000):
    for _ in range(num_iter):
        # Select two random vehicle indices
        selected_vehicle_idx = np.random.randint(0, len(vehicles))
        second_vehicle_idx = np.random.randint(0, len(vehicles))

        # Select the vehicles
        selected_vehicle = vehicles[selected_vehicle_idx]
        second_vehicle = vehicles[second_vehicle_idx]

        # Check if the same vehicle is selected twice
        if selected_vehicle_idx == second_vehicle_idx:
            # Perform 2-opt on a single vehicle
            new_route, new_route_cost = perform_2opt_on_vehicle(selected_vehicle)
            if new_route_cost < selected_vehicle.cost:
                print(f"{Fore.LIGHTMAGENTA_EX}2-Opt Improvement found in single vehicle route {Style.RESET_ALL}{selected_vehicle.ID}")
                selected_vehicle.route = new_route
                selected_vehicle.update_cost()
        else:
            # Perform 2-opt across two different vehicles
            new_route_1, new_route_1_cost, new_route_2, new_route_2_cost = perform_2opt_between_vehicles(selected_vehicle, second_vehicle)
            if new_route_1_cost < selected_vehicle.cost and new_route_2_cost < second_vehicle.cost:
                print(f"{Fore.LIGHTCYAN_EX}2-Opt Cross-vehicle improvement found between vehicles {Style.RESET_ALL}{selected_vehicle.ID} and {second_vehicle.ID}")
                selected_vehicle.route = new_route_1
                second_vehicle.route = new_route_2
                selected_vehicle.update_cost()
                second_vehicle.update_cost()
    return vehicles

def perform_2opt_on_vehicle(vehicle):
    selected_route = vehicle.route
    i, j = np.random.randint(1, len(selected_route) - 1, size=2)
    if j < i:
        i, j = j, i
    
    new_route = selected_route.copy()
    new_route[i:j] = selected_route[j - 1: i - 1: -1]  # Reverse the path between i and j
    
    # Create a temporary vehicle to calculate the new cost
    temp_vehicle = deepcopy(vehicle)
    temp_vehicle.route = new_route
    temp_vehicle.update_cost()
    return new_route, temp_vehicle.cost

def perform_2opt_between_vehicles(vehicle1, vehicle2, num_trials=15):
    best_route1 = vehicle1.route
    best_route2 = vehicle2.route
    best_route1_cost = vehicle1.cost
    best_route2_cost = vehicle2.cost
    best_combined_cost = vehicle1.cost + vehicle2.cost

    for _ in range(num_trials):
        i = np.random.randint(1, len(vehicle1.route) - 1)
        j = np.random.randint(1, len(vehicle2.route) - 1)

        new_route1 = vehicle1.route[:i] + vehicle2.route[j:]
        new_route2 = vehicle2.route[:j] + vehicle1.route[i:]

        # Check if the new routes are within vehicle capacity constraints
        if calculate_route_load(new_route1) <= vehicle1.capacity and calculate_route_load(new_route2) <= vehicle2.capacity:
            # Create temporary vehicles to calculate the new costs
            temp_vehicle1 = deepcopy(vehicle1)
            temp_vehicle2 = deepcopy(vehicle2)
            temp_vehicle1.route = new_route1
            temp_vehicle2.route = new_route2
            temp_vehicle1.update_cost()
            temp_vehicle2.update_cost()
            new_combined_cost = temp_vehicle1.cost + temp_vehicle2.cost

            # Update best routes and costs if the combined cost is improved
            if new_combined_cost < best_combined_cost:
                best_combined_cost = new_combined_cost
                best_route1 = new_route1
                best_route1_cost = temp_vehicle1.cost
                best_route2 = new_route2
                best_route2_cost = temp_vehicle2.cost

    return best_route1, best_route1_cost, best_route2, best_route2_cost

    
def calculate_route_load(route):
    return sum(customer.demand for customer in route)