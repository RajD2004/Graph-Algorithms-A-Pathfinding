
import csv
import heapq
import itertools
import math
import numpy as np
import time
from collections import deque
from typing import Callable, Dict, List, Optional, Set, Tuple, TypeVar


T = TypeVar('T')
Matrix = TypeVar('Matrix')  # Adjacency Matrix
Vertex = TypeVar('Vertex')  # Vertex Class Instance
Graph = TypeVar('Graph')  # Graph Class Instance


class Vertex:
    """ Class representing a Vertex object within a Graph """
    

    __slots__ = ['id', 'adj', 'visited', 'x', 'y']

    def __init__(self, id_init: str, x: float = 0, y: float = 0) -> None:
        """
        DO NOT MODIFY
        Initializes a Vertex
        :param id_init: A unique string identifier used for hashing the vertex
        :param x: The x coordinate of this vertex (used in a_star)
        :param y: The y coordinate of this vertex (used in a_star)
        """
        self.id = id_init
        self.adj = {}  # dictionary {id : weight} of outgoing edges
        self.visited = False  # boolean flag used in search algorithms
        self.x, self.y = x, y  # coordinates for use in metric computations

    def __eq__(self, other: Vertex) -> bool:
        """
        DO NOT MODIFY.
        Equality operator for Graph Vertex class.
        :param other: [Vertex] vertex to compare.
        :return: [bool] True if vertices are equal, else False.
        """
        if self.id != other.id:
            return False
        if self.visited != other.visited:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex visited flags not equal: self.visited={self.visited},"
                  f" other.visited={other.visited}")
            return False
        if self.x != other.x:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex x coords not equal: self.x={self.x}, other.x={other.x}")
            return False
        if self.y != other.y:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex y coords not equal: self.y={self.y}, other.y={other.y}")
            return False
        if set(self.adj.items()) != set(other.adj.items()):
            diff = set(self.adj.items()).symmetric_difference(set(other.adj.items()))
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex adj dictionaries not equal:"
                  f" symmetric diff of adjacency (k,v) pairs = {str(diff)}")
            return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        Constructs string representation of Vertex object.
        :return: [str] string representation of Vertex object.
        """
        lst = [f"<id: '{k}', weight: {v}>" for k, v in self.adj.items()]
        return f"<id: '{self.id}', Adjacencies: {''.join(lst)}>"

    __str__ = __repr__

    def __hash__(self) -> int:
        """
        DO NOT MODIFY
        Hashes Vertex into a set; used in unit tests
        :return: hash value of Vertex
        """
        return hash(self.id)

    # ============== Project 6 Vertex Methods Below ==============#
    def deg(self) -> int:
        """
        Returns the degree (number of outgoing edges from) of this Vertex.
        :return: [int] Degree of this vertex.
        """
        return len(self.adj)

    def get_outgoing_edges(self) -> Set[Tuple[str, float]]:
        """
        Returns a set of tuples (end_id, weight) or empty set if no adjacencies.
        :return: [set[tuple[str, float]]] Set of tuples of the form (end_id, weight).
        """
        return set(self.adj.items())

    def euclidean_distance(self, other: Vertex) -> float:
        """
        Computes the Euclidean distance between this Vertex and another Vertex.
        :param other: [Vertex] Other Vertex to which we wish to compute distance.
        :return: [float] Euclidean distance between this Vertex and another Vertex.
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def taxicab_distance(self, other: Vertex) -> float:
        """
        Returns the taxicab distance between this Vertex and another Vertex.
        :param other: [Vertex] Other Vertex to which we wish to compute distance.
        :return: [float] Taxicab distance between this Vertex and another Vertex.
        """
        return abs(self.x - other.x) + abs(self.y - other.y)


class Graph:
    """ Class implementing the Graph ADT using an Adjacency Map structure """

    __slots__ = ['size', 'vertices', 'plot_show', 'plot_delay']

    def __init__(self, plt_show: bool = False, matrix: Matrix = None, csvf: str = "") -> None:
        """
        DO NOT MODIFY
        Instantiates a Graph class instance
        :param: plt_show : if true, render plot when plot() is called; else, ignore calls to plot()
        :param: matrix : optional matrix parameter used for fast construction
        :param: csvf : optional filepath to a csv containing a matrix
        """
        matrix = matrix if matrix else np.loadtxt(csvf, delimiter=',', dtype=str).tolist() if csvf else None
        self.size = 0
        self.vertices = {}

        self.plot_show = plt_show
        self.plot_delay = 0.2

        if matrix is not None:
            for i in range(1, len(matrix)):
                for j in range(1, len(matrix)):
                    if matrix[i][j] == "None" or matrix[i][j] == "":
                        matrix[i][j] = None
                    else:
                        matrix[i][j] = float(matrix[i][j])
            self.matrix2graph(matrix)

    def __eq__(self, other: Graph) -> bool:
        """
        DO NOT MODIFY
        Overloads equality operator for Graph class
        :param other: graph to compare
        """
        if self.size != other.size or len(self.vertices) != len(other.vertices):
            print(f"Graph size not equal: self.size={self.size}, other.size={other.size}")
            return False
        else:
            for vertex_id, vertex in self.vertices.items():
                other_vertex = other.get_vertex_by_id(vertex_id)
                if other_vertex is None:
                    print(f"Vertices not equal: '{vertex_id}' not in other graph")
                    return False

                adj_set = set(vertex.adj.items())
                other_adj_set = set(other_vertex.adj.items())

                if not adj_set == other_adj_set:
                    print(f"Vertices not equal: adjacencies of '{vertex_id}' not equal")
                    print(f"Adjacency symmetric difference = "
                          f"{str(adj_set.symmetric_difference(other_adj_set))}")
                    return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        :return: String representation of graph for debugging
        """
        return f"Size: {self.size}, Vertices: {list(self.vertices.items())}"

    __str__ = __repr__


    def empty(self) -> bool:
        """
        DO NOT MODIFY
        Checks if the graph is empty
        """
        return self.size == 0


    def plot(self) -> None:
        """
        DO NOT MODIFY
        Creates a plot a visual representation of the graph using matplotlib
        """

        if self.plot_show:
            import matplotlib.cm as cm
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt

            # if no x, y coords are specified, place vertices on the unit circle
            for i, vertex in enumerate(self.get_all_vertices()):
                if vertex.x == 0 and vertex.y == 0:
                    vertex.x = math.cos(i * 2 * math.pi / self.size)
                    vertex.y = math.sin(i * 2 * math.pi / self.size)

            # show edges
            num_edges = len(self.get_all_edges())
            max_weight = max([edge[2] for edge in self.get_all_edges()]) if num_edges > 0 else 0
            colormap = cm.get_cmap('cool')
            for i, edge in enumerate(self.get_all_edges()):
                origin = self.get_vertex_by_id(edge[0])
                destination = self.get_vertex_by_id(edge[1])
                weight = edge[2]

                # plot edge
                arrow = patches.FancyArrowPatch((origin.x, origin.y),
                                                (destination.x, destination.y),
                                                connectionstyle="arc3,rad=.2",
                                                color=colormap(weight / max_weight),
                                                zorder=0,
                                                **dict(arrowstyle="Simple,tail_width=0.5,"
                                                                  "head_width=8,head_length=8"))
                plt.gca().add_patch(arrow)

                # label edge
                plt.text(x=(origin.x + destination.x) / 2 - (origin.x - destination.x) / 10,
                         y=(origin.y + destination.y) / 2 - (origin.y - destination.y) / 10,
                         s=weight, color=colormap(weight / max_weight))

            # show vertices
            x = np.array([vertex.x for vertex in self.get_all_vertices()])
            y = np.array([vertex.y for vertex in self.get_all_vertices()])
            labels = np.array([vertex.id for vertex in self.get_all_vertices()])
            colors = np.array(
                ['yellow' if vertex.visited else 'black' for vertex in self.get_all_vertices()])
            plt.scatter(x, y, s=40, c=colors, zorder=1)

            # plot labels
            for j, _ in enumerate(x):
                plt.text(x[j] - 0.03 * max(x), y[j] - 0.03 * max(y), labels[j])

            # show plot
            plt.show()

            # delay execution to enable animation
            time.sleep(self.plot_delay)

    def add_to_graph(self, begin_id: str, end_id: str = None, weight: float = 1, begin_x_y: Tuple[float, float] = (0, 0), end_x_y: Tuple[float, float] = (0, 0)) -> None:
        """
        Adds to graph: creates start vertex if necessary,
        an edge if specified,
        and a destination vertex if necessary to create said edge
        If edge already exists, update the weight.
        :param begin_id: unique string id of starting vertex
        :param end_id: unique string id of ending vertex
        :param weight: weight associated with edge from start -> dest
        :return: None
        """
        if self.vertices.get(begin_id) is None:
            self.vertices[begin_id] = Vertex(begin_id, *begin_x_y)
            self.size += 1
        if end_id is not None:
            if self.vertices.get(end_id) is None:
                self.vertices[end_id] = Vertex(end_id, *end_x_y)
                self.size += 1
            self.vertices.get(begin_id).adj[end_id] = weight


    def matrix2graph(self, matrix: Matrix) -> None:
        """
        Given an adjacency matrix, construct a graph
        matrix[i][j] will be the weight of an edge between the vertex_ids
        stored at matrix[i][0] and matrix[0][j]
        Add all vertices referenced in the adjacency matrix, but only add an
        edge if matrix[i][j] is not None
        Guaranteed that matrix will be square
        If matrix is nonempty, matrix[0][0] will be None
        :param matrix: an n x n square matrix (list of lists) representing Graph as adjacency map
        :return: None
        """
        for i in range(1, len(matrix)):  # add all vertices to begin with
            self.add_to_graph(matrix[i][0])
        for i in range(1, len(matrix)):  # go back through and add all edges
            for j in range(1, len(matrix)):
                if matrix[i][j] is not None:
                    self.add_to_graph(matrix[i][0], matrix[j][0], matrix[i][j])

    def graph2matrix(self) -> Matrix:
        """
        given a graph, creates an adjacency matrix of the type described in "construct_from_matrix"
        :return: Matrix
        """
        matrix = [[None] + [v_id for v_id in self.vertices]]
        for v_id, outgoing in self.vertices.items():
            matrix.append([v_id] + [outgoing.adj.get(v) for v in self.vertices])
        return matrix if self.size else None

    def graph2csv(self, filepath: str) -> None:
        """
        given a (non-empty) graph, creates a csv file containing data necessary to reconstruct that graph
        :param filepath: location to save CSV
        :return: None
        """
        if self.size == 0:
            return

        with open(filepath, 'w+') as graph_csv:
            csv.writer(graph_csv, delimiter=',').writerows(self.graph2matrix())

    # ============== Graph Methods from Project 6 ==============#

    def unvisit_vertices(self) -> None:
        """
        Resets all visited flags of vertices in Graph to false
        :return: None
        """
        for vertex in self.vertices.values():
            vertex.visited = False

    def get_vertex_by_id(self, vertex_id: str) -> Optional[Vertex]:
        """
        Retrieves a Vertex in the Graph by a provided v_id
        :param vertex_id: unique string id of other vertex
        :return: Vertex object if found; else None
        """
        return self.vertices.get(vertex_id)

    def get_all_vertices(self) -> Set[Vertex]:
        """
        Returns a set of all vertices in the Graph
        :return: set(Vertex)
        """
        return set(self.vertices.values())

    def get_edge_by_ids(self, begin_id: str, end_id: str) -> Optional[Tuple[str, str, float]]:
        """
        Returns the edge connecting the vertex with id begin_id to the vertex with id end_id
        If edge does not exist, returns None
        :return: tuple(begin_id, end_id, weight) or None if edge does not exist
        """
        if self.vertices.get(begin_id) and self.vertices.get(end_id):
            weight = self.vertices.get(begin_id).adj.get(end_id)
            return (begin_id, end_id, weight) if weight is not None else None

    def get_all_edges(self) -> Set[Tuple[str, str, float]]:
        """
        Returns all edges in the graph
        :return: set(tuple(begin_id, end_id, weight)) or empty list if Graph is empty
        """
        return {(begin_id, end_id, v.adj[end_id])
                for begin_id, v in self.vertices.items() for end_id in v.adj}

    def build_path(self, back_edges: Dict[str, str], begin_id: str, end_id: str) -> Tuple[List[str], float]:
        """
        Given a dictionary of back-edges (a mapping of vertex id to predecessor vertex id),
        reconstruct the path from start_id to end_id and compute the total distance
        :param back_edges: Dictionary of back-edges, i.e., (key=vertex_id, value=predecessor_vertex_id) pairs
        :param begin_id: Starting vertex ID string from which to construct path
        :param end_id: Ending vertex ID string to which to construct path
        :return: Tuple where first element is a list of vertex IDs specifying a path from start_id to end_id
                 and the second element is the cost of this path
        """
        path, dist = [end_id], 0
        while path[-1] != begin_id:
            path.append(back_edges.get(path[-1]))  # will construct path in O(V) - no calls to .insert(0)
            dist += self.get_edge_by_ids(path[-1], path[-2])[2]
        return list(reversed(path)), dist

    # ============== Modify Graph Methods Below ==============#
    def bfs(self, begin_id: str, end_id: str) -> Tuple[List[str], float]:
        """
            Performs a breadth-first search (BFS) on the graph to find the shortest path between two vertices.

            This method searches for the shortest path (in terms of the number of edges) from the vertex with
            identifier `begin_id` to the vertex with identifier `end_id`. It returns a tuple containing the
            path as a list of vertex IDs and the total distance (number of edges) along that path.

            :param begin_id: The ID of the starting vertex.
            :param end_id: The ID of the ending vertex.
            :return: A tuple where the first element is a list of vertex IDs representing the shortest path
                    from `begin_id` to `end_id`, and the second element is the total distance of that path.
                    If no path exists, returns ([], 0).
        """
      
        if begin_id == end_id:
            return
        
        self.unvisit_vertices()

        q : deque[str] = deque()
        q.append(begin_id)
        curr_vertex = self.get_vertex_by_id(begin_id)
        end_vertex = self.get_vertex_by_id(end_id)
        if not curr_vertex or not end_vertex:
            return ([],0)
        curr_vertex.visited = True
        backEdges = {}
        while len(q) != 0:
            curr_node = q.popleft()
            curr_node = self.get_vertex_by_id(curr_node)
            curr_neighbors = curr_node.adj #dictionary {id : weight}

            if curr_node.id == end_id:
                break

            for n_id in curr_neighbors.keys():
                n_vertex = self.get_vertex_by_id(n_id)
                if not n_vertex.visited:
                    n_vertex.visited = True
                    backEdges[n_id] = curr_node.id
                    q.append(n_id)
        
        if end_id not in backEdges:
            return ([], 0)
        
        else:
            return self.build_path(back_edges=backEdges, begin_id=begin_id, end_id=end_id)
        

    def a_star(self, begin_id: str, end_id: str,
               metric: Callable[[Vertex, Vertex], float]) -> Tuple[List[str], float]:
        
        """
        Performs the A* search algorithm to find the shortest path between two vertices in the graph.

        The A* algorithm combines features of uniform-cost search and pure heuristic search to efficiently compute
        the shortest path from the starting vertex (`begin_id`) to the ending vertex (`end_id`). It uses a heuristic
        function (`metric`) to estimate the cost from a vertex to the goal.

        :param begin_id: The ID of the starting vertex.
        :param end_id: The ID of the ending vertex.
        :param metric: A callable that takes two vertices and returns a float representing the estimated cost
                    between them (heuristic function h(v)).

        :return: A tuple containing a list of vertex IDs representing the shortest path from `begin_id` to `end_id`,
                and the total cost (distance) of that path. If no path exists, returns ([], 0).
        """

        #f(v) = g(v) + h(v)   h(v) : metric param  g(v) : shortest paths
        beg_vertex, end_vertex = self.get_vertex_by_id(begin_id), self.get_vertex_by_id(end_id)

        if not beg_vertex or not end_vertex:
            return ([],0)
        
        self.unvisit_vertices()

        pq : PriorityQueue[Vertex, float] = PriorityQueue()
        back_edges = {}
        priority = metric(beg_vertex, end_vertex)
        pq.push(priority, beg_vertex)
        beg_vertex.visited = True
        shortest_paths : dict[str, float] = {}
        shortest_paths[beg_vertex.id] = 0

        while not pq.empty():
            curr_weight, curr_node = pq.pop()
            curr_node.visited = True

            if curr_node.id == end_id:
                break
            
            
            for key in curr_node.adj:
                node = self.get_vertex_by_id(key)
                #f(v) = h(v) + g(v)
                path_data = self.get_edge_by_ids(curr_node.id, key)
                path_weight = path_data[2]
                
                g_v = path_weight + shortest_paths.get(curr_node.id)
                if shortest_paths.get(node.id, float('inf')) > g_v:
                    h_v = metric(node, end_vertex)
                    priority_val = g_v + h_v
                    shortest_paths[node.id] = g_v
                    back_edges[node.id] = curr_node.id

                    if node.id in pq.locator:
                        pq.update(priority_val, node)

                    if node.id not in pq.locator:
                        pq.push(priority_val, node)
                        
        
        if end_id not in back_edges:
            return ([], 0)
        
        return self.build_path(back_edges, begin_id, end_id)


def jumanji_path(start_row: int, start_col: int, end_row: int, end_col: int, graph: List[List[int]]) -> Tuple[List[List[int]], float]:
    
    """
    Computes the shortest path in a 2D grid using the A* algorithm.

    The grid consists of free spaces (0) and obstacles (1). The function determines the shortest path from a 
    starting position to an ending position, considering only horizontal and vertical movements.

    :param start_row: The row index of the starting position in the grid.
    :param start_col: The column index of the starting position in the grid.
    :param end_row: The row index of the ending position in the grid.
    :param end_col: The column index of the ending position in the grid.
    :param graph: A 2D list representing the grid. 0 indicates a free space, and 1 indicates an obstacle.

    :return: A tuple containing:
             - A list of coordinates representing the shortest path from the starting position to the ending position.
             - The total distance of the path.
             If no path exists, returns ([], 0).
    """

    new_graph = Graph()
    for i, row in enumerate(graph):
        for j, col in enumerate(row):
            if graph[i][j] == 0:
                i_d = str(i) + "," + str(j)
                new_graph.add_to_graph(begin_id=i_d)

                if i < len(graph) - 1 and graph[i+1][j] == 0:
                    id_3 = str(i+1) + "," + str(j)
                    new_graph.add_to_graph(begin_id=i_d, end_id=id_3, weight=1)

                if i > 0 and graph[i-1][j] == 0:
                    i_d2 = str(i-1) + "," + str(j)
                    new_graph.add_to_graph(begin_id=i_d, end_id=i_d2, weight=1)

                if j < len(row) - 1 and graph[i][j+1] == 0:
                    id_4 = str(i) + "," + str(j+1)
                    new_graph.add_to_graph(begin_id=i_d, end_id=id_4, weight=1)

                if j > 0 and graph[i][j-1] == 0:
                    id_5 = str(i) + ',' + str(j-1)
                    new_graph.add_to_graph(begin_id=i_d, end_id=id_5, weight=1)

    begin_id = str(start_row) + "," + str(start_col)
    end_id = str(end_row) + "," + str(end_col)

    path, distance = new_graph.a_star(begin_id=begin_id, end_id=end_id, metric=Vertex.taxicab_distance)
    if not path:  
        return ([], 0)
    
    formatted_path = [[int(x) for x in node.split(",")] for node in path]
    return (formatted_path, distance)  
    

class Schedule:
    def __init__(self,):
        self.requirements = Graph()

    def isEmpty(self,):
        return self.requirements.empty()
    ########################################
    # MODIFY BELOW #
    ######################################## 

    def addRequirements(self, requirementsToAdd: Dict[str, List[str]]) -> bool:
        
        """
        Adds the specified course requirements to the requirements graph.

        This function checks if the provided course requirements introduce any cycles in the graph.
        If a cycle is detected, the requirements graph is reset to an empty graph, and the function 
        returns False. If no cycle is found, the requirements are added successfully to the graph.

        :param requirementsToAdd: A dictionary where the key is a course (str), and the value is a 
                                list of prerequisite courses (List[str]) required for that course.
        :return: True if the requirements were successfully added without introducing cycles, 
                False otherwise.
        """
            
        def has_cycle(course, visited, rec_stack):
            visited.add(course)
            rec_stack.add(course)

            for prereq in self.requirements.get_vertex_by_id(course).adj:
                if prereq not in visited:  
                    if has_cycle(prereq, visited, rec_stack):
                        return True
                elif prereq in rec_stack:  
                    return True

            rec_stack.remove(course)
            return False

        
        try:
            for course, prereqs in requirementsToAdd.items():
                self.requirements.add_to_graph(course)  
                for prereq in prereqs:
                    self.requirements.add_to_graph(prereq)  
                    self.requirements.add_to_graph(begin_id=prereq, end_id=course, weight=1)  
        except Exception:
            self.requirements = Graph()  
            return False

        
        visited = set()
        rec_stack = set()
        for course in requirementsToAdd.keys():
            if course not in visited:
                if has_cycle(course, visited, rec_stack):
                    self.requirements = Graph()  
                    return False

        return True
    
    def checkSchedule(self, classes: List[List[str]]) -> bool:
        """
        Checks if the given classes form a valid schedule.
        Each inner list in 'classes' represents the classes to be taken in a given semester.

        :param classes: Nested list with each inner list representing a semester.
                        The outer list is sorted in ascending semester order.
        :return: True if the schedule is valid, False otherwise.
        """
        from collections import defaultdict

        prerequisites = defaultdict(set)
        for vertex in self.requirements.vertices.values():
            for course_id in vertex.adj:
                prerequisites[course_id].add(vertex.id)

        taken_classes = set()

        for semester in classes:
            for course in semester:
                if course not in self.requirements.vertices:
                    return False

                course_prereqs = prerequisites.get(course, set())
                if not course_prereqs.issubset(taken_classes):
                    return False

            taken_classes.update(semester)

        return True


########################################
# DO NOT MODIFY BELOW #
######################################## 
class PriorityQueue:
    """
    Priority Queue built upon heapq module with support for priority key updates
    Created by Andrew McDonald
    Inspired by https://docs.python.org/2/library/heapq.html
    """

    __slots__ = ['data', 'locator', 'counter']

    def __init__(self) -> None:
        """
            Construct an AStarPriorityQueue object
            """
        self.data = []  # underlying data list of priority queue
        self.locator = {}  # dictionary to locate vertices within priority queue
        self.counter = itertools.count()  # used to break ties in prioritization

    def __repr__(self) -> str:
        """
            Represent AStarPriorityQueue as a string
            :return: string representation of AStarPriorityQueue object
            """
        lst = [f"[{priority}, {vertex}], " if vertex is not None else "" for
               priority, count, vertex in self.data]
        return "".join(lst)[:-1]

    def __str__(self) -> str:
        """
            Represent AStarPriorityQueue as a string
            :return: string representation of AStarPriorityQueue object
            """
        return repr(self)

    def empty(self) -> bool:
        """
            Determine whether priority queue is empty
            :return: True if queue is empty, else false
            """
        return len(self.data) == 0

    def push(self, priority: float, vertex: Vertex) -> None:
        """
            Push a vertex onto the priority queue with a given priority
            :param priority: priority key upon which to order vertex
            :param vertex: Vertex object to be stored in the priority queue
            :return: None
            """
        # list is stored by reference, so updating will update all refs
        node = [priority, next(self.counter), vertex]
        self.locator[vertex.id] = node
        heapq.heappush(self.data, node)

    def pop(self) -> Tuple[float, Vertex]:
        """
            Remove and return the (priority, vertex) tuple with lowest priority key
            :return: (priority, vertex) tuple where priority is key,
            and vertex is Vertex object stored in priority queue
            """
        vertex = None
        while vertex is None:
            # keep popping until we have valid entry
            priority, count, vertex = heapq.heappop(self.data)
        del self.locator[vertex.id]  # remove from locator dict
        vertex.visited = True  # indicate that this vertex was visited
        while len(self.data) > 0 and self.data[0][2] is None:
            heapq.heappop(self.data)  # delete trailing Nones
        return priority, vertex

    def update(self, new_priority: float, vertex: Vertex) -> None:
        """
            Update given Vertex object in the priority queue to have new priority
            :param new_priority: new priority on which to order vertex
            :param vertex: Vertex object for which priority is to be updated
            :return: None
            """
        node = self.locator.pop(vertex.id)  # delete from dictionary
        node[-1] = None  # invalidate old node
        self.push(new_priority, vertex)  # push new node



