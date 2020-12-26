'''
Functions for 3D Processing
'''

# Imports
import numpy as np
import open3d
import panda3d
from direct.showbase import ShowBase
from direct.showbase.Loader import Loader

# Main Functions
def DepthImage_to_Terrain(depths, I, ImagePath, name='Test', exportPath=None):
    vertexTextureUVs = [[0.0, 0.0]]
    vertexNormals = [[0.0, 1.0, 0.0]]
    
    vertices = []
    for i in range(depths.shape[0]):
        for j in range(depths.shape[1]):
            vertices.append([j+1, i+1, depths[i, j]])

    faceMaps = []
    for i in range(depths.shape[0]-1):
        for j in range(depths.shape[1]-1):
            faceMaps.append([
                [(i*depths.shape[1]) + j +1, 1, 1], 
                [((i+1)*depths.shape[1]) + j +1, 1, 1], 
                [((i+1)*depths.shape[1]) + j+1 +1, 1, 1], 
                [(i*depths.shape[1]) + j+1 +1, 1, 1]
            ])

    mesh = Object3D(name, vertices, faceMaps, vertexTextureUVs, vertexNormals)
    
    if not (exportPath is None):
        # Initial Write Without Normals
        OBJWrite(mesh, exportPath)

        # Reread and Compute Normals and ReExport
        mesh = open3d.io.read_triangle_mesh(exportPath)
        mesh.compute_vertex_normals()
    
        open3d.visualization.draw_geometries([mesh])
        open3d.io.write_triangle_mesh(exportPath, mesh)
        
        loader = Loader(ShowBase)
        model = loader.loadModel(exportPath)
        # model.reparentTo(panda3d.render)
        
        tex = loader.loadTexture(ImagePath)
        model.setTexture(tex, 1)
    
    return mesh

def Points_to_3DModel(points, colors, method='poisson', exportPath=None, displayMesh=True):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    pcd.estimate_normals()
    print(np.asarray(pcd.normals))

    if displayMesh:
        open3d.visualization.draw_geometries([pcd])

    final_mesh = None
    if method == 'Poisson':
        # Poisson Reconstruction
        final_mesh = PoissonReconstruction(pcd)
    else:
        # Ball Rolling Algo
        final_mesh = BallRollingAlgo(pcd)

    # Export Mesh
    if exportPath is not None:
        open3d.io.write_triangle_mesh(exportPath, final_mesh)

    return final_mesh

def BallRollingAlgo(pcd):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 30 * avg_dist

    bpa_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, open3d.utility.DoubleVector([radius, radius * 2]))
    # Decimate to lower triangles
    # bpa_mesh = bpa_mesh.simplify_quadric_decimation(100000)
    # Clean Mesh
    bpa_mesh.remove_degenerate_triangles()
    bpa_mesh.remove_duplicated_triangles()
    bpa_mesh.remove_duplicated_vertices()
    bpa_mesh.remove_non_manifold_edges()

    return bpa_mesh

def PoissonReconstruction(pcd):
    poisson_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)
    return p_mesh_crop

# OBJ Writer Functions
class Object3D:
    def __init__(self, name, vertices, faceMaps, vertexTextureUVs=None, vertexNormals=None):
        self.name = name
        self.vertices = vertices
        self.faceMaps = faceMaps
        self.vertexTextureUVs = vertexTextureUVs
        self.vertexNormals = vertexNormals
        if self.vertexTextureUVs is None:
            self.vertexTextureUVs = [[0, 0, 0]]
        if self.vertexNormals is None:
            self.vertexNormals = [[0, 0, 0]]


def OBJWrite(obj, savePath):
    """
    # Blender v2.81 (sub 16) OBJ File: ''
    # www.blender.org
    mtllib Cube.mtl
    o Cube
    v 0.000000 0.000000 0.000000
    v 1.000000 0.000000 0.000000
    v 1.000000 1.000000 0.000000
    v 0.000000 1.000000 0.000000
    vt 0.625000 0.500000
    vt 0.875000 0.500000
    vt 0.875000 0.750000
    vt 0.625000 0.750000
    vn 0.0000 1.0000 0.0000
    vn 0.0000 0.0000 1.0000
    vn -1.0000 0.0000 0.0000
    vn 0.0000 -1.0000 0.0000
    usemtl Material
    s off
    f 1/1/1 2/2/2 3/3/3 4/4/4
    """
    Header = [
    "# Blender v2.81 (sub 16) OBJ File: ''",
    '# www.blender.org',
    'mtllib Cube.mtl'
    ]

    Name = 'o ' + obj.name

    Vertices = []
    for v in obj.vertices:
        Vertices.append('v ' + Get6DecFloatString(v[0]) + ' ' + Get6DecFloatString(v[1]) + ' ' + Get6DecFloatString(v[2]))
    
    VertexTextureUVs = []
    for v in obj.vertexTextureUVs:
        VertexTextureUVs.append('vt ' + Get6DecFloatString(v[0]) + ' ' + Get6DecFloatString(v[1]))
    
    VertexNormals = []
    for v in obj.vertexNormals:
        VertexNormals.append('vn ' + Get6DecFloatString(v[0]) + ' ' + Get6DecFloatString(v[1]) + ' ' + Get6DecFloatString(v[2]))
    
    MidText = [
        'usemtl Material',
        's off'
    ]

    Faces = []
    for f in obj.faceMaps:
        fline = 'f '
        for v in f:
            fline = fline + '' + str(v[0]) + '/' + str(v[1]) + '/' + str(v[2]) + ' '
        Faces.append(fline.rstrip())
    
    OBJTextLines = list(Header)
    OBJTextLines.append(Name)
    OBJTextLines.extend(Vertices)
    OBJTextLines.extend(VertexTextureUVs)
    OBJTextLines.extend(VertexNormals)
    OBJTextLines.extend(MidText)
    OBJTextLines.extend(Faces)

    open(savePath, 'w').write('\n'.join(OBJTextLines))

def Get6DecFloatString(val):
    val = round(float(val), 6)
    decCount = len(str(val).split('.')[1])
    extraPadding = 6 - decCount
    strval = str(val) + ('0'*extraPadding)
    return strval

# Driver Code
# depths = np.array([
#     [0, 0], 
#     [0, 0]
# ])
# colors = np.array([])
# exportPath = 'TestImgs/Testt.obj'
# DepthImage_to_Terrain(depths, colors, 'Test', exportPath)