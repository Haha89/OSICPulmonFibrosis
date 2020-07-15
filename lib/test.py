# -*- coding: utf-8 -*-

# import tools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tools


PATH_DATA = "../data/"


# =============================================================================
# get volume of lungs
# =============================================================================

def make_mesh(image, threshold=0.3, step_size=1):

    print("Transposing surface")
    p = image.transpose(2,1,0)
    
    print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces


def plt_3d(verts, faces):
    print("Drawing")
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.xaxis.pane.set_edgecolor('b')
    ax.yaxis.pane.set_edgecolor('b')
    ax.zaxis.pane.set_edgecolor('b')
    #ax.set_color_cycle((0.7, 0.7, 0.7))
    plt.show()

id = "ID00026637202179561894768"
processed_mat = tools.get_data_patient(id_patient="ID00007637202177411956430")
verts, faces = make_mesh(processed_mat)
plt_3d(verts, faces)