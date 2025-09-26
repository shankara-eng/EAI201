# campus_graph.py
"""
Campus graph (adjacency list with weights in meters) and normalized coordinates.
Normalized coordinates (x,y) are relative to the full image size (0..1).
If you resize the map image, the normalized coords will still locate nodes correctly.

Edit distances here if you have measured values.
"""

# Adjacency list: graph[node] = {neighbor: distance_m, ...}
# Undirected graph: each connection is present in both directions.
campus_graph = {
    "Main Gate": {"Library": 54, "Acad 1": 30},
    "Library": {"Main Gate": 54, "Acad 1": 40, "Cafe": 30},
    "Acad 1": {"Main Gate": 30, "Library": 40, "Cafe": 24, "Acad 2": 37},
    "Cafe": {"Library": 30, "Acad 1": 24, "Acad 2": 120},
    "Acad 2": {"Cafe": 120, "Acad 1": 37, "J1": 76},
    "J1": {"Acad 2": 76, "J2": 73, "Food Court": 128},
    "J2": {"J1": 73, "Food Court": 55, "Faculty Housing": 240},
    "Food Court": {"J1": 128, "J2": 55, "Hostel": 202},
    "Hostel": {"Food Court": 202, "Mart": 55, "Faculty Housing": 240},
    "Mart": {"Hostel": 55},
    "Faculty Housing": {"Hostel": 240, "J2": 240, "J3": 325},
    "J3": {"Faculty Housing": 325, "Sports": 325},
    "Sports": {"J3": 325, "Cricket": 150},
    "Cricket": {"Sports": 150}
}

# Normalized coordinates (x,y) between 0 and 1 relative to the image width/height.
# If you open the UI and dots are slightly off, tweak these values.
coords_norm = {
    # main circle / bottom area
    "Main Gate": (0.28, 0.92),
    "Library": (0.34, 0.86),
    "Acad 1": (0.45, 0.78),
    "Cafe": (0.42, 0.74),
    "Acad 2": (0.38, 0.62),

    # middle/right area
    "J1": (0.40, 0.68),
    "J2": (0.55, 0.55),
    "Food Court": (0.48, 0.52),
    "Hostel": (0.70, 0.40),
    "Mart": (0.76, 0.38),
    "Faculty Housing": (0.65, 0.52),

    # top area / sports
    "J3": (0.58, 0.30),
    "Sports": (0.42, 0.20),
    "Cricket": (0.35, 0.14),
}

# lists (for dropdowns / coloring)
LOCATIONS = ["Main Gate", "Library", "Acad 1", "Cafe", "Acad 2",
             "Hostel", "Mart", "Faculty Housing", "Food Court",
             "Sports", "Cricket"]
JUNCTIONS = ["J1", "J2", "J3"]

# Map scale used by A* heuristic: this is "meters per full-image normalized unit".
# Default is a guess (1000 m). You should calibrate it for more accurate A* behaviour.
# See instructions in README comment below on how to calibrate this.
MAP_SCALE_METERS = 1000.0

# helper: get all nodes
def get_all_nodes():
    return list(campus_graph.keys())

# =========================
# How to calibrate MAP_SCALE_METERS:
# If you know the real-world distance (meters) between two nodes A and B,
# you can set MAP_SCALE_METERS so the heuristic uses realistic meters:
#
#   normalized_dist = sqrt((xA-xB)^2 + (yA-yB)^2)
#   # then choose MAP_SCALE_METERS such that:
#   MAP_SCALE_METERS = real_distance_m / normalized_dist
#
# Example (python REPL):
#   from campus_graph import coords_norm
#   import math
#   nd = math.hypot(coords_norm['Hostel'][0]-coords_norm['Mart'][0],
#                   coords_norm['Hostel'][1]-coords_norm['Mart'][1])
#   map_scale = 55 / nd
#   print(map_scale)
#
# Then copy that number into MAP_SCALE_METERS above.
# =========================
