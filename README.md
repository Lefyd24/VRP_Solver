# VRP Optimization Assignment - Solution using Local Search
> #### MSc in Business Analytics - AUEB (2023-2024)

> #### Course: Large Scale Optimization

This repository contains the solution to the Vehicle Routing Problem (VRP) Optimization Assignment for the course of Large Scale Optimization. 

Three move types are used for local search:
- **Swap**: Swap two nodes between two routes or within the same route
- **2-opt**: Swap two edges between two routes or within the same route
- **Relocate**: Move a node from one route to another, or from one position to another in the same route

The neighborhood types are then fed inside a VND (Variable Neighborhood Descent) algorithm, which is a metaheuristic for solving optimization problems. The algorithm starts with a random solution and then iteratively moves to the best solution in the neighborhood. The algorithm stops when no better solution is found in the neighborhood.

> The solution can be found on the requested format inside the folder `solutions/`, while also a corresponding plot of the vehicle routes of the solution can be found inside the folder `plots/`.