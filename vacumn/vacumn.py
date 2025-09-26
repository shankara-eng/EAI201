import tkinter as tk
import math

# ----- User chooses shape -----
shape_choice = input("Enter shape (circle/square/hexagon/pentagon): ").strip().lower()

# ----- Main window -----
window = tk.Tk()
window.title("Vacuum Cleaner Simulator")
window.geometry("600x600")

canvas = tk.Canvas(window, width=600, height=600, bg="white")
canvas.pack()

# ----- Initial state -----
x, y = 300, 300   # position (center)
size = 40
angle = 90        # facing "up" initially (degrees)
shape_id = None
move_speed = 5
moving = False

# ----- Function to draw rotated polygons -----
def draw_shape(shape, x, y, size, angle):
    canvas.delete("all")  # clear previous drawing

    if shape == "circle":
        return canvas.create_oval(x-size, y-size, x+size, y+size, fill="lightblue")

    elif shape == "square":
        points = [
            (-size, -size), (size, -size),
            (size, size), (-size, size)
        ]
    elif shape == "hexagon":
        points = []
        for i in range(6):
            a = math.radians(60 * i)
            points.append((size * math.cos(a), size * math.sin(a)))
    elif shape == "pentagon":
        points = []
        for i in range(5):
            a = math.radians(72 * i - 90)
            points.append((size * math.cos(a), size * math.sin(a)))
    else:
        return None

    # Rotate points
    rotated = []
    for px, py in points:
        rad = math.radians(angle)
        rx = px * math.cos(rad) - py * math.sin(rad)
        ry = px * math.sin(rad) + py * math.cos(rad)
        rotated.append(x + rx)
        rotated.append(y + ry)

    return canvas.create_polygon(rotated, fill="lightgreen" if shape == "square" else 
                                 "lightpink" if shape == "hexagon" else 
                                 "lightyellow")

# Draw chosen shape
shape_id = draw_shape(shape_choice, x, y, size, angle)

# ----- Movement Functions -----
def move_forward():
    global x, y, moving, shape_id
    if moving:
        # Move in direction of current angle
        rad = math.radians(angle)
        dx = move_speed * math.cos(rad)
        dy = move_speed * math.sin(rad)

        # Boundary check
        if 0 < x+dx < 600 and 0 < y+dy < 600:
            x += dx
            y += dy
            shape_id = draw_shape(shape_choice, x, y, size, angle)

        window.after(50, move_forward)

def start(event):
    global moving
    moving = True
    move_forward()

def stop_and_exit(event):
    window.destroy()

def turn_left(event):
    global angle, shape_id
    angle -= 15   # rotate 15° left
    shape_id = draw_shape(shape_choice, x, y, size, angle)

def turn_right(event):
    global angle, shape_id
    angle += 15   # rotate 15° right
    shape_id = draw_shape(shape_choice, x, y, size, angle)

def dock(event):
    """Move shape smoothly back to center (300, 300)."""
    global x, y, shape_id
    dx = 300 - x
    dy = 300 - y

    if abs(dx) > 2 or abs(dy) > 2:
        x += dx / 20
        y += dy / 20
        shape_id = draw_shape(shape_choice, x, y, size, angle)
        window.after(50, lambda: dock(event))

# ----- Bind keys -----
window.bind("<s>", start)        # Start moving
window.bind("<S>", start)
window.bind("<x>", stop_and_exit)  # Exit
window.bind("<X>", stop_and_exit)
window.bind("<Left>", turn_left)   # Rotate left
window.bind("<Right>", turn_right) # Rotate right
window.bind("<d>", dock)           # Dock
window.bind("<D>", dock)

# ----- Run -----
window.mainloop()
