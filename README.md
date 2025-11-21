# Graph Algorithms, A* Search, BFS, Jumanji Grid Pathing, and Course Scheduling

This project implements a full graph ADT using adjacency maps, plus several algorithms for pathfinding and prerequisite checking.

## Features

### 🔧 Graph Structure
- `Vertex` class with:
  - adjacency list
  - visited flag
  - coordinates (for heuristics)
  - distance metrics (Euclidean + Taxicab)

- `Graph` class supporting:
  - Adding vertices/edges  
  - Converting between adjacency matrix ↔ graph  
  - Getting edges, vertices, IDs  
  - Utility methods (`unvisit_vertices`, `build_path`, etc.)

### 🔍 Pathfinding Algorithms
- **BFS**  
  Finds the shortest unweighted path between two vertices.  
  Returns `(path, distance)`.

- **A\* Search**  
  Supports any heuristic (uses Euclidean or Taxicab).  
  Efficient weighted shortest path.

- **PriorityQueue**  
  Custom queue with locator-based priority updates.

### 🎮 Jumanji Grid Path
`jumanji_path()` converts a 2D grid (0 = free, 1 = obstacle) into a graph and uses A* to compute the shortest valid path.

Returns:
([ [row, col], ... ], distance)


### 🎓 Schedule Checking
`Schedule` class validates course prerequisites using a directed acyclic graph:
- `addRequirements()`  
  Adds prerequisites; rejects + resets if they create a cycle.

- `checkSchedule()`  
  Confirms semester-by-semester course ordering is valid.

### 📄 File
- `solution.py` — full implementation including Graph, Vertex, BFS, A*, Jumanji pathing, and Schedule logic. :contentReference[oaicite:0]{index=0}

## Summary
A complete graph toolkit used for pathfinding, grid navigation, and validating course prerequisite structures.
