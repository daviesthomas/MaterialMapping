import bpy
from blender.material import PrincipledBSDF, Color
import numpy as np

class Geometry():
    def __init__(self, filepath, rotation_euler=(0.0,0.0,0.0), normalize = True):
        x = rotation_euler[0] * 1.0 / 180.0 * np.pi 
        y = rotation_euler[1] * 1.0 / 180.0 * np.pi 
        z = rotation_euler[2] * 1.0 / 180.0 * np.pi 
        angle = (x,y,z)
        
        prev = [obj.name for obj in bpy.data.objects]

        if ('.stl' in filepath or '.STL' in filepath):
            bpy.ops.import_mesh.stl(filepath=filepath)
        elif ('.obj' in filepath or '.OBJ' in filepath):
            bpy.ops.import_scene.obj(filepath=filepath, split_mode='OFF')
        else:
            raise(Exception("Unkown mesh format! Must be obj or stl!"))

        after = [obj.name for obj in bpy.data.objects]

        # new name is diff 
        name = list(set(after) - set(prev))[0]

        self.mesh = bpy.data.objects[name]
        self.mesh.location = (0.0, 0.0, 0.0)
        self.mesh.rotation_euler = angle

        if (normalize):
            maxDim = max(self.mesh.dimensions)
            self.mesh.dimensions = self.mesh.dimensions/maxDim

        # place on ground plane
        self.__moveToGround()

        # just default to smooth shading
        bpy.ops.object.shade_smooth()
        bpy.context.view_layer.update()
        
    def setBSDF(self, bsdf: PrincipledBSDF):
        mat = bpy.data.materials.new('MeshMaterial')
        self.mesh.data.materials.append(mat)
        self.mesh.active_material = mat
        mat.use_nodes = True
        tree = mat.node_tree

        # set principled BSDF
        bsdf.setNode(tree.nodes["Principled BSDF"])

    def subdivide(self, level):
        bpy.context.view_layer.objects.active = self.mesh
        bpy.ops.object.modifier_add(type='SUBSURF')

        # set subdivsion in both render and 3d views
        self.mesh.modifiers["Subdivision"].render_levels = level 
        self.mesh.modifiers["Subdivision"].levels = level

    def __moveToGround(self):
        # get the scene
        scene = bpy.context.scene

        # set geometry to origin
        bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN")

        zverts = []

        # get all z coordinates of the vertices
        for face in self.mesh.data.polygons:
            verts_in_face = face.vertices[:]
            for vert in verts_in_face:
                local_point = self.mesh.data.vertices[vert].co
                world_point = self.mesh.matrix_world @ local_point
                zverts.append(world_point[2])

        # set the minimum z coordinate as z for cursor location
        scene.cursor.location = (0, 0, min(zverts))

        # set the origin to the cursor
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

        # set the object to (0,0,0)
        self.mesh.location = (0,0,0)

        # reset the cursor
        scene.cursor.location = (0,0,0)

    def delete(self):
        bpy.ops.object.delete()


    