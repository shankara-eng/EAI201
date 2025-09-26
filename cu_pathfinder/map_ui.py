# map_ui.py
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import math

from campus_graph import campus_graph, coords_norm, LOCATIONS, JUNCTIONS, get_all_nodes, MAP_SCALE_METERS
import pathfinding as pf

# File name of your satellite image (must be in same folder or give full path)
IMAGE_PATH = "cu_satpic.jpg"

# Window size
WIN_W = 1200
WIN_H = 800
SIDEBAR_W = int(WIN_W * 0.20)  # 20% for left side

# drawing parameters
NODE_RADIUS = 6
LOCATION_COLOR = "orange"
JUNCTION_COLOR = "green"
EDGE_COLOR = "#0b66c2"  # blue for base path lines
HIGHLIGHT_COLOR = "red"

# estimate walking speed: 5 km/h -> 83.333... m/min
WALKING_SPEED_M_PER_MIN = 5000.0 / 60.0

class CampusNavigator:
    def __init__(self, root):
        self.root = root
        self.root.title("CU PATH FINDER")
        self.root.geometry(f"{WIN_W}x{WIN_H}")

        # sidebar frame (left)
        self.sidebar = tk.Frame(root, width=SIDEBAR_W, bg="#f4f4f4")
        self.sidebar.pack(side="left", fill="y")
        self._build_sidebar()

        # map frame (right)
        self.map_frame = tk.Frame(root, bg="white")
        self.map_frame.pack(side="right", expand=True, fill="both")

        # Canvas
        self.canvas = tk.Canvas(self.map_frame, bg="white")
        self.canvas.pack(fill="both", expand=True)

        # load and display image
        self._load_image(IMAGE_PATH)

        # state for drawn items so we can clear highlights
        self.edge_items = []    # base edges
        self.node_items = {}    # node -> (oval_id, text_id)
        self.highlight_items = []  # current highlighted path items

        # draw base map overlays (nodes + edges)
        self.draw_edges()
        self.draw_nodes()

        # enable panning with mouse drag
        self.canvas.bind("<ButtonPress-1>", self._on_button_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        # mouse wheel for zoom (Windows / Mac / Linux differences)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", self._on_mousewheel)    # some linux
        self.canvas.bind("<Button-5>", self._on_mousewheel)

        # remember last mouse position for pan
        self._drag_data = {"x": 0, "y": 0}

    def _build_sidebar(self):
        tk.Label(self.sidebar, text="CU PATH FINDER", font=("Arial", 16, "bold"), bg="#f4f4f4").pack(pady=12)

        tk.Label(self.sidebar, text="Start:", bg="#f4f4f4").pack(anchor="w", padx=10)
        self.start_var = tk.StringVar()
        start_cb = ttk.Combobox(self.sidebar, textvariable=self.start_var, values=get_all_nodes())
        start_cb.pack(fill="x", padx=10, pady=6)

        tk.Label(self.sidebar, text="Destination:", bg="#f4f4f4").pack(anchor="w", padx=10)
        self.end_var = tk.StringVar()
        end_cb = ttk.Combobox(self.sidebar, textvariable=self.end_var, values=get_all_nodes())
        end_cb.pack(fill="x", padx=10, pady=6)

        tk.Label(self.sidebar, text="Algorithm:", bg="#f4f4f4").pack(anchor="w", padx=10)
        self.alg_var = tk.StringVar()
        alg_cb = ttk.Combobox(self.sidebar, textvariable=self.alg_var,
                              values=["Breadth-First Search", "Depth-First Search",
                                      "Uniform Cost Search", "A* Search"])
        alg_cb.pack(fill="x", padx=10, pady=6)
        alg_cb.current(0)

        tk.Button(self.sidebar, text="Find Path", command=self.on_find_path).pack(pady=12, padx=10, fill="x")

        # info label
        self.info_text = tk.Text(self.sidebar, height=10, wrap="word")
        self.info_text.pack(padx=10, pady=6, fill="both", expand=True)

        # zoom and pan controls
        ctrl_frame = tk.Frame(self.sidebar, bg="#f4f4f4")
        ctrl_frame.pack(side="bottom", pady=8, padx=8, fill="x")
        tk.Button(ctrl_frame, text="+ Zoom", command=lambda: self.zoom(1.2)).pack(side="left", expand=True, fill="x")
        tk.Button(ctrl_frame, text="- Zoom", command=lambda: self.zoom(1/1.2)).pack(side="left", expand=True, fill="x")
        tk.Button(ctrl_frame, text="Reset view", command=self.reset_view).pack(side="left", expand=True, fill="x")

    def _load_image(self, path):
        # load with PIL and put on canvas
        pil_img = Image.open(path)
        # fit image to canvas area (map area): target width = WIN_W - SIDEBAR_W
        target_w = WIN_W - SIDEBAR_W
        # maintain aspect ratio
        w, h = pil_img.size
        scale = min(target_w / w, WIN_H / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.img_w, self.img_h = new_w, new_h
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(pil_img)
        # We create a group at (0,0)
        self.canvas_image = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        # make canvas scrollable area match image
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))

    def norm_to_pixel(self, node):
        """Convert normalized coords to pixel coords on current image."""
        if node not in coords_norm:
            raise KeyError(f"No coords for node '{node}' in coords_norm. Add it in campus_graph.py")
        nx, ny = coords_norm[node]
        x = int(nx * self.img_w)
        y = int(ny * self.img_h)
        return x, y

    def draw_nodes(self):
        """Draw small dots and labels for all nodes."""
        # remove existing node items first
        for v in self.node_items.values():
            for iid in v:
                try: self.canvas.delete(iid)
                except: pass
        self.node_items.clear()

        for node, (nx, ny) in coords_norm.items():
            x, y = self.norm_to_pixel(node)
            color = LOCATION_COLOR if node in LOCATIONS else JUNCTION_COLOR
            oid = self.canvas.create_oval(x - NODE_RADIUS, y - NODE_RADIUS, x + NODE_RADIUS, y + NODE_RADIUS,
                                          fill=color, outline="black", width=1)
            tid = self.canvas.create_text(x + 12, y, text=node, anchor="w", font=("Arial", 10, "bold"), fill=color)
            self.node_items[node] = (oid, tid)

    def draw_edges(self):
        """Draw lines for each edge once (undirected)."""
        # remove existing base edges first
        for eid in self.edge_items:
            try: self.canvas.delete(eid)
            except: pass
        self.edge_items = []

        seen = set()
        for a, neighbors in campus_graph.items():
            for b in neighbors.keys():
                key = tuple(sorted([a, b]))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    x1, y1 = self.norm_to_pixel(a)
                    x2, y2 = self.norm_to_pixel(b)
                except KeyError:
                    continue
                eid = self.canvas.create_line(x1, y1, x2, y2, fill=EDGE_COLOR, width=3)
                self.edge_items.append(eid)

    def clear_highlight(self):
        for iid in self.highlight_items:
            try: self.canvas.delete(iid)
            except: pass
        self.highlight_items = []

    def on_find_path(self):
        start = self.start_var.get()
        goal = self.end_var.get()
        alg = self.alg_var.get()
        if not start or not goal:
            messagebox.showwarning("Input missing", "Choose both start and destination.")
            return
        if start not in campus_graph or goal not in campus_graph:
            messagebox.showerror("Node error", "Start or destination not known in graph.")
            return

        self.clear_highlight()
        result = None
        if alg == "Breadth-First Search":
            result = pf.bfs(campus_graph, start, goal)
        elif alg == "Depth-First Search":
            result = pf.dfs(campus_graph, start, goal)
        elif alg == "Uniform Cost Search":
            result = pf.ucs(campus_graph, start, goal)
        elif alg == "A* Search":
            # pass coords_norm and map scale to A*
            result = pf.astar(campus_graph, start, goal, coords_norm=coords_norm, map_scale_meters=MAP_SCALE_METERS)
        else:
            messagebox.showerror("Algorithm", "Unknown algorithm selected.")
            return

        self._display_result(result)
        if result and result.get("path"):
            self.highlight_path(result["path"])

    def _display_result(self, result):
        self.info_text.delete("1.0", "end")
        if not result or not result.get("path"):
            self.info_text.insert("end", "No path found.\n")
            return
        path = result["path"]
        dist = result["distance"]
        explored = result["explored"]
        minutes = dist / WALKING_SPEED_M_PER_MIN
        mins = int(minutes)
        secs = int((minutes - mins) * 60)
        info = f"Path: {'  →  '.join(path)}\nDistance: {dist:.1f} m\nEstimated time: {mins} min {secs} sec\nNodes explored: {explored}\n"
        self.info_text.insert("end", info)

    def highlight_path(self, path):
        # draw thick red polyline and highlight nodes
        # node circles drawn earlier; create bigger circles for path nodes
        px = [self.norm_to_pixel(node) for node in path]
        # draw lines between path nodes
        for i in range(len(px)-1):
            x1, y1 = px[i]
            x2, y2 = px[i+1]
            lid = self.canvas.create_line(x1, y1, x2, y2, fill=HIGHLIGHT_COLOR, width=6)
            self.highlight_items.append(lid)
        # highlight nodes
        for node in path:
            x, y = self.norm_to_pixel(node)
            r = NODE_RADIUS + 3
            nid = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=HIGHLIGHT_COLOR, outline="")
            self.highlight_items.append(nid)

    # ---------- simple pan / zoom ----------
    def _on_button_press(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _on_mouse_drag(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def zoom(self, factor):
        # zoom around center of canvas
        cx = self.canvas.winfo_width() / 2
        cy = self.canvas.winfo_height() / 2
        self.canvas.scale("all", cx, cy, factor, factor)
        # update scrollregion
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.config(scrollregion=bbox)

    def _on_mousewheel(self, event):
        # Windows: event.delta positive (wheel up) / negative (wheel down)
        if hasattr(event, "delta"):
            if event.delta > 0:
                self.zoom(1.1)
            else:
                self.zoom(1/1.1)
        else:
            # some linux: Button-4 = up, Button-5 = down
            if event.num == 4:
                self.zoom(1.1)
            else:
                self.zoom(1/1.1)

    def reset_view(self):
        # reload image and redraw everything to reset transforms
        self.canvas.delete("all")
        self._load_image(IMAGE_PATH)
        self.draw_edges()
        self.draw_nodes()
        self.highlight_items = []
        self.edge_items = []

if __name__ == "__main__":
    root = tk.Tk()
    app = CampusNavigator(root)
    root.mainloop()
