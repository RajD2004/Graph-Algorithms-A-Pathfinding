
import math, random, unittest, string
from solution import Graph, Vertex, jumanji_path, Schedule

class GraphTests(unittest.TestCase):

    def test_bfs(self):
        graph = Graph()

        # (1) test on empty graph
        subject = graph.bfs('a', 'b')
        self.assertEqual(([], 0), subject)

        # (2) test on graph missing begin or dest
        graph.add_to_graph('a')
        subject = graph.bfs('a', 'b')
        self.assertEqual(([], 0), subject)
        subject = graph.bfs('b', 'a')
        self.assertEqual(([], 0), subject)

        # (3) test on graph with both vertices but no path
        graph.add_to_graph('b')
        subject = graph.bfs('a', 'b')
        self.assertEqual(([], 0), subject)

        # (4) test on single edge
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        subject = graph.bfs('a', 'b')
        self.assertEqual((['a', 'b'], 331), subject)

        # (5) test on two edges
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        graph.add_to_graph('b', 'c', 100)
        subject = graph.bfs('a', 'c')
        self.assertEqual((['a', 'b', 'c'], 431), subject)

        # (6) test on edge triangle and ensure one-edge path is taken
        # (bfs guarantees fewest-edge path, not least-weighted path)
        graph = Graph()
        graph.add_to_graph('a', 'b', 331)
        graph.add_to_graph('b', 'c', 100)
        graph.add_to_graph('a', 'c', 999)
        subject = graph.bfs('a', 'c')
        self.assertEqual((['a', 'c'], 999), subject)

        # (7) test on grid figure-8 and ensure fewest-edge path is taken
        graph = Graph(csvf='test_csvs/bfs/7.csv')

        subject = graph.bfs('bottomleft', 'topleft')
        self.assertEqual((['bottomleft', 'midleft', 'topleft'], 2), subject)

        graph.unvisit_vertices()  # mark all unvisited
        subject = graph.bfs('bottomright', 'topright')
        self.assertEqual((['bottomright', 'midright', 'topright'], 2), subject)

        graph.unvisit_vertices()  # mark all unvisited
        subject = graph.bfs('bottomleft', 'topright')
        self.assertIn(subject[0], [['bottomleft', 'midleft', 'topleft', 'topright'],
                                    ['bottomleft', 'midleft', 'midright', 'topright'],
                                    ['bottomleft', 'bottomright', 'midright', 'topright']])
        self.assertEqual(3, subject[1])

        # (8) test on example graph from Onsay's slides, starting from vertex A
        # see bfs_graph.png
        graph = Graph(csvf='test_csvs/bfs/8.csv')

        subject = graph.bfs('a', 'd')
        self.assertEqual((['a', 'b', 'd'], 4), subject)

        graph.unvisit_vertices()  # mark all unvisited
        subject = graph.bfs('a', 'f')
        self.assertEqual((['a', 'c', 'f'], 4), subject)

        graph.unvisit_vertices()  # mark all unvisited
        subject = graph.bfs('a', 'h')
        self.assertEqual((['a', 'e', 'h'], 4), subject)

        graph.unvisit_vertices()  # mark all unvisited
        subject = graph.bfs('a', 'g')
        self.assertEqual((['a', 'e', 'g'], 4), subject)

        graph.unvisit_vertices()  # mark all unvisited
        subject = graph.bfs('a', 'i')
        self.assertIn(subject[0], [['a', 'e', 'h', 'i'], ['a', 'e', 'g', 'i']])
        self.assertEqual(6, subject[1])

        # (9) test path which does not exist
        graph.unvisit_vertices()  # mark all unvisited
        graph.add_to_graph('z')
        subject = graph.bfs('a', 'z')
        self.assertEqual(([], 0), subject)

    def test_a_star(self):

        # PART ONE -- SMALLER TEST CASES

        # === Edge Cases === #

        # (1) test on empty graph
        graph = Graph()
        subject = graph.a_star('a', 'b', lambda v1, v2: 0)
        self.assertEqual(([], 0), subject)

        # (2) start/end vertex does not exist
        graph = Graph()
        graph.add_to_graph('a')
        # (2.1) start vertex
        subject = graph.a_star('b', 'a', lambda v1, v2: 0)
        self.assertEqual(([], 0), subject)
        # (2.2) end vertex
        subject = graph.a_star('a', 'b', lambda v1, v2: 0)
        self.assertEqual(([], 0), subject)
        # (2.3) Neither vertex exists (Also tested in 3)
        subject = graph.a_star('b', 'c', lambda v1, v2: 0)
        self.assertEqual(([], 0), subject)

        # (3) test for path which does not exist
        graph = Graph()
        graph.add_to_graph('a', 'b')
        subject = graph.a_star('b', 'a', lambda v1, v2: 0)
        self.assertEqual(([], 0), subject)

        # === (A) Grid graph tests ===#
        graph = Graph()

        # (1) test on nxn grid from corner to corner: should shoot diagonal
        # (shortest path is unique, so each heuristic will return the same path)
        grid_size = 5
        for x in range(grid_size):
            for y in range(grid_size):
                idx = f'{x},{y}'
                graph.vertices[idx] = Vertex(idx, x, y)
                graph.size += 1

        for x in range(grid_size):
            for y in range(grid_size):
                if x < grid_size - 1:
                    graph.add_to_graph(f'{x},{y}', f'{x + 1},{y}', 1)
                    graph.add_to_graph(f'{x + 1},{y}', f'{x},{y}', 1)
                if y < grid_size - 1:
                    graph.add_to_graph(f'{x},{y}', f'{x},{y + 1}', 1)
                    graph.add_to_graph(f'{x},{y + 1}', f'{x},{y}', 1)
                if x < grid_size - 1 and y < grid_size - 1:
                    graph.add_to_graph(f'{x},{y}', f'{x + 1},{y + 1}', math.sqrt(2))
                    graph.add_to_graph(f'{x + 1},{y + 1}', f'{x},{y}', math.sqrt(2))

        for metric in [Vertex.euclidean_distance, Vertex.taxicab_distance]:
            subject = graph.a_star('0,0', '4,4', metric)
            self.assertEqual(['0,0', '1,1', '2,2', '3,3', '4,4'], subject[0])
            self.assertAlmostEqual((grid_size - 1) * math.sqrt(2), subject[1])
            graph.unvisit_vertices()

        # (2) test on nxn grid with penalty for shooting diagonal
        # (shortest path is not unique, so each heuristic will return a different path)
        for x in range(grid_size - 1):
            for y in range(grid_size - 1):
                graph.add_to_graph(f'{x},{y}', f'{x + 1},{y + 1}', 3)
                graph.add_to_graph(f'{x + 1},{y + 1}', f'{x},{y}', 3)

        subject = graph.a_star('0,0', '4,4', Vertex.euclidean_distance)
        self.assertEqual((['0,0', '1,0', '1,1', '2,1', '2,2', '3,2', '3,3', '4,3', '4,4'], 8), subject)
        graph.unvisit_vertices()
        subject = graph.a_star('0,0', '4,4', Vertex.taxicab_distance)
        self.assertEqual((['0,0', '1,0', '2,0', '3,0', '4,0', '4,1', '4,2', '4,3', '4,4'], 8), subject)
        graph.unvisit_vertices()

        # === (B) Tollway graph tests ===#
        graph = Graph(csvf='test_csvs/astar/tollway_comprehensive_2.csv')
        # now must set of coordinates for each vertex:
        positions = [(0, 0), (2, 0), (4, 0), (7, 0), (10, 0), (12, 0), (2, 5), (6, 4), (12, 5), (5, 9), (8, 8), (12, 8),
                     (8, 10), (0, 2),
                     (4, 2), (9, 2), (9, -2), (7, 6), (8, 11), (14, 8)]

        for index, v_id in enumerate(list(graph.vertices)):
            graph.vertices[v_id].x, graph.vertices[v_id].y = positions[index]

        # UNCOMMENT TO SEE PLOT
        # graph.plot_show = True
        # graph.plot()

        for metric in [Vertex.euclidean_distance, Vertex.taxicab_distance]:
            # (3) test Franklin Grove to Northbrook shortest path in both directions
            # (shortest path is unique, so each heuristic will return the same path)
            subject = graph.a_star('Franklin Grove', 'Northbrook', metric)
            solution = (['Franklin Grove', 'A', 'B', 'G', 'J', 'M', 'Northbrook'], 22)
            self.assertEqual(solution, subject)
            graph.unvisit_vertices()

            subject = graph.a_star('Northbrook', 'Franklin Grove', metric)
            solution = (['Northbrook', 'M', 'J', 'G', 'B', 'A', 'Franklin Grove'], 22)
            self.assertEqual(solution, subject)
            graph.unvisit_vertices()

            # (4) test Franklin Grove to Joliet shortest path - bypass expensive tollway path
            # (shortest path is unique, so each heuristic will return the same path)
            subject = graph.a_star('Franklin Grove', 'Joliet', metric)
            solution = (['Franklin Grove', 'A', 'B', 'G', 'H', 'D', 'E', 'Joliet'], 35)
            self.assertEqual(solution, subject)
            graph.unvisit_vertices()

            subject = graph.a_star('Joliet', 'Franklin Grove', metric)
            solution = (['Joliet', 'E', 'D', 'H', 'G', 'B', 'A', 'Franklin Grove'], 35)
            self.assertEqual(solution, subject)
            graph.unvisit_vertices()

            # (5) test Joliet to Chicago shortest path - bypass expensive tollway path
            # (shortest path is unique, so each heuristic will return the same path)
            subject = graph.a_star('Joliet', 'Chicago', metric)
            solution = (['Joliet', 'E', 'D', 'H', 'G', 'J', 'K', 'L', 'Chicago'], 35)
            self.assertEqual(solution, subject)
            graph.unvisit_vertices()

            subject = graph.a_star('Chicago', 'Joliet', metric)
            solution = (['Chicago', 'L', 'K', 'J', 'G', 'H', 'D', 'E', 'Joliet'], 35)
            self.assertEqual(solution, subject)
            graph.unvisit_vertices()

            # (6) test Northbrook to Belvidere - despite equal path lengths, A* heuristic will always prefer search to the left
            # (both heuristics will prefer the same path)
            subject = graph.a_star('Northbrook', 'Belvidere', metric)
            solution = (['Northbrook', 'M', 'J', 'K', 'Belvidere'], 8)
            self.assertEqual(solution, subject)
            graph.unvisit_vertices()

            subject = graph.a_star('Belvidere', 'Northbrook', metric)
            solution = (['Belvidere', 'K', 'J', 'M', 'Northbrook'], 8)
            self.assertEqual(solution, subject)
            graph.unvisit_vertices()

        # PART 2 -- BIGGER TEST CASES

        # === (C) Random graph tests ===#
        # (1) initialize vertices of Euclidean and Taxicab weighted random graphs
        random.seed(331)
        probability = 0.5  # probability that two vertices are connected
        e_graph, t_graph = Graph(), Graph()
        vertices = []
        for s in string.ascii_lowercase:
            x, y = random.randint(0, 100), random.randint(0, 100)
            vertex = Vertex(s, x, y)
            vertices.append(vertex)
            e_graph.vertices[s], t_graph.vertices[s] = vertex, vertex
            e_graph.size += 1
            t_graph.size += 1

        # (2) construct adjacency matrix with edges weighted by appropriate distance metric
        e_matrix = [[None] + [s for s in string.ascii_lowercase]]
        t_matrix = [[None] + [s for s in string.ascii_lowercase]]
        for i in range(1, len(e_matrix[0])):
            e_row = [e_matrix[0][i]]
            t_row = [t_matrix[0][i]]
            for j in range(1, len(e_matrix[0])):
                connect = (random.random() < probability)  # connect if random draw in (0,1) < probability
                e_weighted_dist, t_weighted_dist = None, None
                if i != j and connect:
                    e_dist = vertices[i - 1].euclidean_distance(vertices[j - 1])
                    t_dist = vertices[i - 1].taxicab_distance(vertices[j - 1])
                    weight = (random.randint(1, 10))  # choose a random weight between 1 and 9
                    e_weighted_dist = e_dist * weight  # create realistic weighted dist
                    t_weighted_dist = t_dist * weight  # create realistic weighted dist
                e_row.append(e_weighted_dist)
                t_row.append(t_weighted_dist)
            e_matrix.append(e_row)
            t_matrix.append(t_row)
        e_graph.matrix2graph(e_matrix)
        t_graph.matrix2graph(t_matrix)

        # (3) define helper function to check validity of search result
        def is_valid_path(graph, search_result):
            path, dist = search_result
            length = 0
            for i in range(len(path) - 1):
                begin, end = path[i], path[i + 1]
                edge = graph.get_edge_by_ids(begin, end)
                if edge is None:
                    return False  # path contains some edge not in the graph
                length += edge[2]
            return length == dist  # path consists of valid edges: return whether length matches

        # (4) test all 26 x 26 pairwise A* traversals across random matrix and ensure they return valid paths w/o error
        for begin in vertices:
            for end in vertices:
                if begin != end:
                    subject = e_graph.a_star(begin.id, end.id, Vertex.euclidean_distance)
                    self.assertTrue(is_valid_path(e_graph, subject))
                    e_graph.unvisit_vertices()

                    subject = t_graph.a_star(begin.id, end.id, Vertex.taxicab_distance)
                    self.assertTrue(is_valid_path(t_graph, subject))
                    t_graph.unvisit_vertices()

    def test_jumanji(self):
        start_row = 0
        start_col = 1
        end_row = 4
        end_col = 3
        graph = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ]
        expected = [[0, 1], [0, 0], [1, 0], [2, 0], [2, 1], [3, 1], [4, 1], [4, 2], [4, 3]]

        path, distance = jumanji_path(start_row, start_col, end_row, end_col, graph)
        self.assertEqual(expected, path)
        self.assertEqual(8, distance)

        path, distance = jumanji_path(end_row, end_col, start_row, start_col, graph)
        expected.reverse()
        self.assertEqual(expected, path)
        self.assertEqual(8, distance)

        graph = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        start = (0, 0)
        end = (3, 3)
        path, distance = jumanji_path(*start , *end, graph)
        self.assertEqual([], path)
        self.assertEqual(0, distance)

        start = (0, 0)
        end = (1, 3)
        path, distance = jumanji_path(*start , *end, graph)
        pathSouth = [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3]]
        self.assertEqual(pathSouth, path)
        self.assertEqual(4, distance)

        graph = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
        start = (0, 0)
        end = (4, 4)
        path, distance = jumanji_path(*start , *end, graph)
        self.assertEqual([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4], [4, 4]], path)
        self.assertEqual(8, distance)

        # Vertical path has precedence (Top -> Bottom)
        graph[2][0] = 0
        start = (0, 0)
        end = (4, 4)
        path, distance = jumanji_path(*start , *end, graph)
        self.assertEqual([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]], path)
        self.assertEqual(8, distance)

        graph = [
            [0, 0],
            [0, 0]
        ]
        start = (0, 0)
        end = (1, 1)
        path, distance = jumanji_path(*start , *end, graph)
        self.assertEqual([[0, 0], [1, 0], [1, 1]], path)
        self.assertEqual(2, distance)

        # Vertical path has precedence (bottom->top)
        path, distance = jumanji_path(*end , *start, graph)
        self.assertEqual([[1, 1], [0, 1], [0, 0]], path)
        self.assertEqual(2, distance)

        graph = [
            [0, 1],
            [1, 0]
        ]
        start = (0, 0)
        end = (1, 1)
        path, distance = jumanji_path(*start , *end, graph)
        self.assertEqual([], path)
        self.assertEqual(0, distance)

        graph = [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ]
        start = (0, 0)
        end = (4, 0)
        path, distance = jumanji_path(*start , *end, graph)
        self.assertEqual([[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [4, 1], [4, 0]], path)
        self.assertEqual(8, distance)

        # Vertical path has precedence and south path is preferred
        graph = [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
        start = (1, 0)
        end = (4, 4)
        path, distance = jumanji_path(*start , *end, graph)
        self.assertEqual([[1,0], [2,0], [3,0], [4,0], [5,0], [5,1], [5,2], [5,3], [5,4],  [4,4]], path)
        self.assertEqual(9, distance)

class ScheduleTests(unittest.TestCase):
    def test_add_requirements(self,):
        # Test 0 : Valid - Empty requirements
        r = { }
        s = Schedule()
        self.assertTrue(s.addRequirements(r))
        self.assertTrue(s.isEmpty())

        # Test 1 : Valid - Simple requirements
        r = {
            'CSE 231': [], #CSE 231 has no requirements, it is a freshman class
            'CSE 232': ['CSE 231'],
            'CSE 331': ['CSE 232'],
        }
        s = Schedule()
        self.assertTrue(s.addRequirements(r))
        self.assertFalse(s.isEmpty())

        # Test 2: Simple schedule with same class cycle
        r = {
            'CSE 231': ['CSE 231'],
        }
        s = Schedule()
        self.assertFalse(s.addRequirements(r))
        self.assertTrue(s.isEmpty())

        # Test 3 : Invalid - Contains direct cycle
        r = {
            'CSE 231': [],
            'CSE 232': ['CSE 331'],
            'CSE 331': ['CSE 232'], # Easy cycle here
        }
        s = Schedule()
        self.assertFalse(s.addRequirements(r))
        self.assertTrue(s.isEmpty())


        # Test 4 : Valid - All classes have no requirements
        r = {
            'CSE 102': [],
            'MTH 103': [],
            'MTH 102': [],
        }
        s = Schedule()
        self.assertTrue(s.addRequirements(r))
        self.assertFalse(s.isEmpty())

        # Test 5 : Valid - Only one class
        r = {
            'CSE 102': []
        }
        s = Schedule()
        self.assertTrue(s.addRequirements(r))  
        self.assertFalse(s.isEmpty())

        # Test 6 : Valid - Bigger schedule
        r = {
            'MTH 132': [],
            'MTH 133': ['MTH 132'],
            'MTH 234': ['MTH 133'],
            'CSE 231': ['MTH 132'],
            'CSE 232': ['CSE 231'],
            'CSE 260': ['MTH 132'],
            'CSE 300': [],
            'CSE 331': ['CSE 260', 'CSE 232'],
            'CSE 335': ['CSE 331'],
            'CSE 404': ['CSE 331', 'MTH 234'],
            'CSE 498': ['CSE 300', 'CSE 335', 'MTH 234']
        }
        s = Schedule()
        self.assertTrue(s.addRequirements(r))
        self.assertFalse(s.isEmpty())

        # Test 7 : Invalid - Contains indirect cycle
        r = {
            'CSE 231': ['CSE 331'],
            'CSE 232': ['CSE 231'],
            'CSE 331': ['CSE 232'],
        }
        s = Schedule()
        self.assertFalse(s.addRequirements(r))
        self.assertTrue(s.isEmpty())

        # Test 8 : Invalid - Big schedule with cycle
        r = {
            'MTH 132': ['CSE 232'],
            'MTH 133': ['MTH 132'],
            'MTH 234': ['MTH 133'],
            'CSE 231': ['MTH 132'],
            'CSE 232': ['CSE 231'],
            'CSE 260': ['MTH 132'],
            'CSE 300': [],
            'CSE 331': ['CSE 260', 'CSE 232', 'CSE 498'],
            'CSE 335': ['CSE 331'],
            'CSE 404': ['CSE 331', 'MTH 234'],
            'CSE 498': ['CSE 300', 'CSE 335', 'MTH 234']
        }
        s = Schedule()
        self.assertFalse(s.addRequirements(r))
        self.assertTrue(s.isEmpty())

    def test_checkSchedule(self,):
        r = {
            'CSE 231': [], #CSE 231 has no requirements, it is a freshman class
            'CSE 232': ['CSE 231'],
            'CSE 331': ['CSE 232'],
        }

        s = Schedule()
        self.assertTrue(s.addRequirements(r))
        studentSchedule = [
            ['CSE 231'], ['CSE 232'], ['CSE 331']
        ]
        self.assertTrue(s.checkSchedule(studentSchedule)) # Returns true
        
        s.requirements.unvisit_vertices()
        wrongSchedule = [
            ['CSE 232'], ['CSE 231'], ['CSE 331']
        ]
        self.assertFalse(s.checkSchedule(wrongSchedule)) # Returns false

        r = {
            'CSE 102': [],
            'MTH 103': [],
            'MTH 102': [],
        }
        s = Schedule()
        s.addRequirements(r)
        validSchedule = [
            ['CSE 102'], ['MTH 103']
        ]
        self.assertTrue(s.checkSchedule(validSchedule))
        s.requirements.unvisit_vertices()

        validSchedule = [
            ['MTH 103'], ['CSE 102']
        ]
        self.assertTrue(s.checkSchedule(validSchedule))

        # Test 5 : Valid - Bigger schedule
        r = {
            'MTH 132': [],
            'MTH 133': ['MTH 132'],
            'MTH 234': ['MTH 133'],
            'CSE 231': ['MTH 132'],
            'CSE 232': ['CSE 231'],
            'CSE 260': ['MTH 132'],
            'CSE 300': [],
            'CSE 331': ['CSE 260', 'CSE 232'],
            'CSE 335': ['CSE 331'],
            'CSE 404': ['CSE 331', 'MTH 234'],
            'CSE 498': ['CSE 300', 'CSE 335', 'MTH 234']
        }
        s = Schedule()
        s.addRequirements(r)

        validSchedule = [
            ['MTH 132', 'CSE 300'], 
            ['CSE 231', 'MTH 133', 'CSE 260'], 
            ['CSE 232', 'MTH 234'],
            ['CSE 331', ],
            ['CSE 335', 'CSE 404'],
            ['CSE 498']
        ]
        self.assertTrue(s.checkSchedule(validSchedule))
        s.requirements.unvisit_vertices()

        invalidSchedule = [
            ['MTH 132', 'CSE 300', 'CSE 231'], 
            ['MTH 133', 'CSE 260'], 
            ['CSE 232', 'MTH 234'],
            ['CSE 331', ],
            ['CSE 335', 'CSE 404'],
            ['CSE 498']
        ]
        self.assertFalse(s.checkSchedule(invalidSchedule))
        s.requirements.unvisit_vertices()

        invalidSchedule = [
            ['MTH 132', 'CSE 300'], 
            ['CSE 231', 'MTH 133', 'CSE 260', 'CSE 232'], 
            ['MTH 234'],
            ['CSE 331', ],
            ['CSE 335', 'CSE 404'],
            ['CSE 498']
        ]
        self.assertFalse(s.checkSchedule(invalidSchedule))
        s.requirements.unvisit_vertices()

        invalidSchedule = [
            ['MTH 132', 'CSE 300'], 
            ['CSE 231', 'MTH 133', 'CSE 260', 'CSE 404'], 
            ['CSE 232', 'MTH 234'],
            ['CSE 331'],
            ['CSE 335'],
            ['CSE 498']
        ]
        self.assertFalse(s.checkSchedule(invalidSchedule))
        s.requirements.unvisit_vertices()


if __name__ == '__main__':
    unittest.main()
