import matplotlib.pyplot as plt
import numpy as np 


# Points on the object expressed in the b-frame
points_b = np.array([[0, 0, 0],
                    [0.3, 0, 0],
                    [0.3, 0, 0.15],
                    [0, 0, 0.15],
                    [0, 0.1, 0],
                    [0.3, 0.1, 0],
                    [0.3, 0.1, 0.15],
                    [0, 0.1, 0.15]])

# Lines on the cuboid as sequence of tuples containing the indices of the starting point and the endpoint
edges = [[0, 1], [1, 2], [2, 3], [3, 0],  # Lines of front plane
         [4, 5], [5, 6], [6, 7], [7, 4],  # Lines of back plane
         [0, 4], [1, 5], [2, 6], [3, 7]]  # Lines connecting front with back-plane


def plot_edges(image_points, edges):
    plt.figure()
    for edge in edges:
        pt1 = image_points[edge[0], :]
        pt2 = image_points[edge[1], :]
        x1 = pt1[0]
        x2 = pt2[0]
        y1 = pt1[1]
        y2 = pt2[1]
        # in the plot y points upwards so i use -y to let positive y point downwards in the plot
        plt.plot([x1, x2], [-y1, -y2], 'b') 
    plt.title("Perspective view of cuboid")
    plt.show()
    return


# To test the plot_image function, the 3d points are plotted by ignoring their z-component. This is an orthographic projection
image_points = points_b[:,:2]
plot_edges(image_points, edges)





