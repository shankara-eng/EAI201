# pathfinding.py
import heapq
import math

def _path_distance(graph, path):
    if not path or len(path) == 1:
        return 0
    dist = 0
    for i in range(len(path)-1):
        a, b = path[i], path[i+1]
        dist += graph[a][b]
    return dist

def bfs(graph, start, goal):
    from collections import deque
    queue = deque([[start]])
    visited = set()
    nodes_explored = 0
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return {"path": path, "distance": _path_distance(graph, path), "explored": nodes_explored}
        if node not in visited:
            visited.add(node)
            nodes_explored += 1
            for neigh in graph.get(node, {}):
                if neigh not in visited:
                    queue.append(path + [neigh])
    return {"path": None, "distance": None, "explored": nodes_explored}

def dfs(graph, start, goal):
    stack = [[start]]
    visited = set()
    nodes_explored = 0
    while stack:
        path = stack.pop()
        node = path[-1]
        if node == goal:
            return {"path": path, "distance": _path_distance(graph, path), "explored": nodes_explored}
        if node not in visited:
            visited.add(node)
            nodes_explored += 1
            for neigh in graph.get(node, {}):
                if neigh not in visited:
                    stack.append(path + [neigh])
    return {"path": None, "distance": None, "explored": nodes_explored}

def ucs(graph, start, goal):
    pq = []
    heapq.heappush(pq, (0, [start]))
    best_cost = {start: 0}
    nodes_explored = 0
    while pq:
        cost, path = heapq.heappop(pq)
        node = path[-1]
        if node == goal:
            return {"path": path, "distance": cost, "explored": nodes_explored}
        if cost > best_cost.get(node, float("inf")):
            continue
        nodes_explored += 1
        for neigh, w in graph[node].items():
            new_cost = cost + w
            if new_cost < best_cost.get(neigh, float("inf")):
                best_cost[neigh] = new_cost
                heapq.heappush(pq, (new_cost, path + [neigh]))
    return {"path": None, "distance": None, "explored": nodes_explored}

def _heuristic(a, b, coords_norm):
    x1, y1 = coords_norm[a]
    x2, y2 = coords_norm[b]
    return math.hypot(x1 - x2, y1 - y2) * 1000  # scale factor

def astar(graph, start, goal, coords_norm):
    pq = []
    start_h = _heuristic(start, goal, coords_norm)
    heapq.heappush(pq, (start_h, 0, [start]))
    best_g = {start: 0}
    nodes_explored = 0
    while pq:
        f, g, path = heapq.heappop(pq)
        node = path[-1]
        if node == goal:
            return {"path": path, "distance": g, "explored": nodes_explored}
        if g > best_g.get(node, float("inf")):
            continue
        nodes_explored += 1
        for neigh, w in graph[node].items():
            new_g = g + w
            if new_g < best_g.get(neigh, float("inf")):
                best_g[neigh] = new_g
                h = _heuristic(neigh, goal, coords_norm)
                heapq.heappush(pq, (new_g + h, new_g, path + [neigh]))
    return {"path": None, "distance": None, "explored": nodes_explored}
