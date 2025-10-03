import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- 1. Load Pre-trained Model and Setup Environment ---

MODEL_PATH = 'saved_model'

# Check if the saved model exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at '{MODEL_PATH}'")
    print("Please run 'train_model.py' first to train and save the model.")
    exit()

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 2. Define the Maze and Test Scenario ---

# This layout must match the one used for training
maze_layout = [
    [2, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
]

maze = np.array(maze_layout, dtype=np.float32)
start_pos = tuple(np.argwhere(maze == 2)[0])
end_pos = tuple(np.argwhere(maze == 3)[0])

# --- 3. Visualization Function ---

def visualize_path(maze_to_draw, path, title="Evacuation Route"):
    """Uses Matplotlib to draw the maze, threats, and the calculated path."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title, fontsize=16)

    colors = {0: 'white', 1: 'black', 2: 'green', 3: 'blue', 4: 'red'}
    
    rows, cols = maze_to_draw.shape
    ax.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])

    for r in range(rows):
        for c in range(cols):
            cell_type = maze_to_draw[r, c]
            ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=colors.get(cell_type, 'white')))
            
            if cell_type == 2: ax.text(c, r, 'S', ha='center', va='center', color='white', fontsize=20)
            if cell_type == 3: ax.text(c, r, 'E', ha='center', va='center', color='white', fontsize=20)
            if cell_type == 4: ax.text(c, r, 'X', ha='center', va='center', color='white', fontsize=20, weight='bold')

    if path:
        path_x = [c for r, c in path]
        path_y = [r for r, c in path]
        ax.plot(path_x, path_y, marker='o', markersize=10, color='orange', linestyle='-', linewidth=3, label="Predicted Path")

    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()


# --- 4. Main Execution ---

if __name__ == "__main__":
    print("\n--- Testing the Model on a New Scenario ---")
    
    # Create a test maze with specific threat locations
    test_maze = np.copy(maze)
    # YOU CAN CHANGE THE THREAT LOCATIONS HERE TO TEST DIFFERENT SCENARIOS
    test_maze[1, 8] = 4
    test_maze[4, 5] = 4
    test_maze[8, 2] = 4
    
    # Prepare the maze for the model
    input_for_prediction = np.expand_dims(test_maze, axis=0)
    
    # Get the model's prediction
    predicted_path_flat = model.predict(input_for_prediction)[0]
    predicted_path_grid = predicted_path_flat.reshape(maze.shape)

    # --- Path Reconstruction ---
    # Extract coordinates where the model's confidence is above a threshold
    path_coords_raw = np.argwhere(predicted_path_grid > 0.5)
    
    path_coords = []
    if len(path_coords_raw) > 0:
        current_pos = start_pos
        remaining_points = [tuple(p) for p in path_coords_raw]

        while current_pos != end_pos and len(remaining_points) > 0:
            path_coords.append(current_pos)
            if current_pos in remaining_points:
                remaining_points.remove(current_pos)
            
            # Find the next closest point from the model's prediction
            distances = [np.linalg.norm(np.array(current_pos) - np.array(p)) for p in remaining_points]
            if not distances:
                break
            
            next_idx = np.argmin(distances)
            current_pos = remaining_points.pop(next_idx)

        if end_pos not in path_coords:
             path_coords.append(end_pos)
    else:
        print("Model could not find a clear path.")

    # Visualize the final result
    visualize_path(test_maze, path_coords, title="ANN-Predicted Evacuation Route")
