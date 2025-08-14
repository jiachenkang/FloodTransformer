import netCDF4 as nc
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

DATA_PATH = "data/val/221092.nc"
DEVICE = torch.device('cuda:2')
DTYPE = torch.bfloat16

def get_vertices(data_path=DATA_PATH, grid_size=5):
    """
    Draw squares defined by vertex coordinates
    
    Parameters:
    x_coords: array of shape (N, 4), containing x coordinates of N squares
    y_coords: array of shape (N, 4), containing y coordinates of N squares
    """
    with nc.Dataset(data_path, mode='r') as dataset:
        x_coords = dataset.variables['Mesh2DContour_x'][:]
        y_coords = dataset.variables['Mesh2DContour_y'][:]
    # Adjust coordinates to origin
    min_x = 520700 / grid_size# np.min(x_coords)
    min_y = 6104100 / grid_size #6104250 / grid_size# np.min(y_coords)
    
    x_adjusted = x_coords / grid_size - min_x
    y_adjusted = y_coords / grid_size - min_y
    
    # Create vertex array
    vertices = np.dstack((x_adjusted, y_adjusted))

    # Print some statistics
    print(f"Original coordinate range:")
    print(f"X: [{np.min(x_coords):.2f}, {np.max(x_coords):.2f}]")
    print(f"Y: [{np.min(y_coords):.2f}, {np.max(y_coords):.2f}]")
    print(f"\nAdjusted coordinate range:")
    print(f"X: [{np.min(x_adjusted):.2f}, {np.max(x_adjusted):.2f}]")
    print(f"Y: [{np.min(y_adjusted):.2f}, {np.max(y_adjusted):.2f}]")

    min_x_filter = 0 / grid_size
    max_x_filter = 39300 / grid_size

    min_vals = vertices[:, :, 0].min(axis=1)
    max_vals = vertices[:, :, 0].max(axis=1)

    valid_indices = (min_vals >= min_x_filter) & (min_vals < max_x_filter)

    filtered_vertices = vertices[valid_indices]
    print("\nfiltered_vertices.shape", filtered_vertices.shape)
    print(f"Coordinate range:")
    print(f"X: [{np.min(filtered_vertices[:,:,0]):.2f}, {np.max(filtered_vertices[:,:,0]):.2f}]")
    print(f"Y: [{np.min(filtered_vertices[:,:,1]):.2f}, {np.max(filtered_vertices[:,:,1]):.2f}]")

    return filtered_vertices, valid_indices


def get_water_level(valid_indices, data_path=DATA_PATH):
    with nc.Dataset(data_path, mode='r') as dataset:
        water_level = dataset.variables['Mesh2D_s1'][:]
    return water_level[:,valid_indices]


def plot_vertices(vertices, figsize=(7,15), value=None, time_step=None):
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create polygon collection
    if value is None:
        poly = PolyCollection(
            vertices,
            facecolors=np.column_stack([np.full_like(vertices[:,0,1], 1), np.full_like(vertices[:,0,1], 1), vertices[:,0,1]/3490]),
            edgecolors='blue',
            linewidth=0.1,
            alpha=0.8
        )
    else:
        poly = PolyCollection(
            vertices,
            array=value[time_step],
            cmap='viridis',
            edgecolors='face',
            linewidth=0
        )
    
    # Add polygons to figure
    ax.add_collection(poly)
    
    # Set axis range
    ax.set_xlim(np.min(vertices[:, :, 0]), np.max(vertices[:, :, 0]))
    ax.set_ylim(np.min(vertices[:, :, 1]), np.max(vertices[:, :, 1]))
    
    # Remove all borders and axes
    ax.set_frame_on(False)  # Remove figure border
    ax.set_xticks([])      # Remove x-axis ticks
    ax.set_yticks([])      # Remove y-axis ticks
    
    # Keep aspect ratio equal
    ax.set_aspect('equal')
    
    # Show figure
    plt.show()



def plot_data_distribution(value):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    original_value = value.reshape(-1)
    # Linear scale histogram
    ax1.hist(original_value, bins=100)
    ax1.set_title('Linear Scale Histogram')
    ax1.set_xlabel('Pixel Value') 
    ax1.set_ylabel('Frequency')

    # Log scale histogram
    ax2.hist(original_value, bins=100)
    ax2.set_yscale('log')
    ax2.set_title('Log Scale Histogram')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency (log)')

    plt.tight_layout()
    plt.savefig('output_image.png')  # Save image to file
    plt.close() 


def vertices_to_grid(vertices, values, grid_shape=(3490, 7860)):

    # Create empty output array
    grid_data = np.zeros(grid_shape)
    
    # Fill values into corresponding grid positions
    for i in range(len(values)):
        # Get square boundaries
        min_x = int(np.min(vertices[i, :, 0]))
        max_x = int(np.max(vertices[i, :, 0]))
        min_y = int(np.min(vertices[i, :, 1]))
        max_y = int(np.max(vertices[i, :, 1]))
        
        grid_data[np.maximum(0, min_y):np.minimum(grid_shape[0], max_y), 
                  min_x:np.minimum(grid_shape[1], max_x)] = values[i]
    
    return np.flipud(grid_data)

def plot_data(data, figsize=(15, 7), save_path='output.png'):
    plt.figure(figsize=figsize)
    plt.imshow(data, cmap='viridis')
    plt.axis('off')
    plt.savefig(save_path)  # Save image to file
    plt.close()  # Close figure window


def compute_adjacency_matrix(contour_x, contour_y, tol=1e-3):
    """
    Compute adjacency matrix for multi-scale square grid cells using contour coordinates
    
    Args:
        contour_x: np.array [N, 4] - x coordinates of four vertices for each cell, e.g. Mesh2DContour_x
        contour_y: np.array [N, 4] - y coordinates of four vertices for each cell, e.g. Mesh2DContour_y
        tol: float - tolerance for boundary alignment
    
    Returns:
        adjacency_matrix: np.array [N, 4] where each row contains indices of
            [left, right, top, bottom] neighbors (-1 = no neighbor)
    """
    N = contour_x.shape[0]
    adjacency_matrix = -np.ones((N, 4), dtype=int)
    
    # For each cell, get its boundaries
    # Since squares are axis-aligned, we can sort x and y coordinates
    x_bounds = np.sort(contour_x, axis=1)  # left and right x coordinates
    y_bounds = np.sort(contour_y, axis=1)  # bottom and top y coordinates
    
    # Get unique x and y coordinates for each cell [N]
    left = x_bounds[:, 0]   # leftmost x 
    right = x_bounds[:, -1] # rightmost x 
    bottom = y_bounds[:, 0] # bottom y 
    top = y_bounds[:, -1]   # top y 


    for i in range(N):
        for j in range(i + 1, N):
            # Check vertical adjacency (left/right)
            if abs(right[i] - left[j]) < tol:
                # Calculate vertical overlap
                overlap = min(top[i], top[j]) - max(bottom[i], bottom[j])
                if overlap > tol:
                    adjacency_matrix[i, 1] = j  # j is right neighbor of i
                    adjacency_matrix[j, 0] = i  # i is left neighbor of j
                    
            elif abs(right[j] - left[i]) < tol:
                overlap = min(top[i], top[j]) - max(bottom[i], bottom[j])
                if overlap > tol:
                    adjacency_matrix[i, 0] = j  # j is left neighbor of i
                    adjacency_matrix[j, 1] = i  # i is right neighbor of j
                    
            # Check horizontal adjacency (top/bottom)
            if abs(top[i] - bottom[j]) < tol:
                # Calculate horizontal overlap
                overlap = min(right[i], right[j]) - max(left[i], left[j])
                if overlap > tol:
                    adjacency_matrix[i, 2] = j  # j is top neighbor of i
                    adjacency_matrix[j, 3] = i  # i is bottom neighbor of j
                    
            elif abs(top[j] - bottom[i]) < tol:
                overlap = min(right[i], right[j]) - max(left[i], left[j])
                if overlap > tol:
                    adjacency_matrix[i, 3] = j  # j is bottom neighbor of i
                    adjacency_matrix[j, 2] = i  # i is top neighbor of j
                    
    return adjacency_matrix
