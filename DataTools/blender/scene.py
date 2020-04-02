import bpy
import numpy as np

class Scene():
    def __init__(self, castShadows = False):
        if castShadows:
            self.invisibleGround(location = (0,0,0), groundSize = 20, shadowBrightness = 0.7)
        
    def invisibleGround(self, location = (0,0,0), groundSize = 20, shadowBrightness = 0.7):
        # initialize a ground for shadow
        bpy.context.scene.cycles.film_transparent = True
        bpy.ops.mesh.primitive_plane_add(location = location, size = groundSize)
        bpy.context.object.cycles.is_shadow_catcher = True

        # # set material
        ground = bpy.context.object
        mat = bpy.data.materials.new('MeshMaterial')
        ground.data.materials.append(mat)
        mat.use_nodes = True
        tree = mat.node_tree
        tree.nodes["Principled BSDF"].inputs['Transmission'].default_value = shadowBrightness

    def setAmbientLight(self, color = (0,0,0,1)):
        bpy.data.scenes[0].world.use_nodes = True
        bpy.data.scenes[0].world.node_tree.nodes["Background"].inputs['Color'].default_value = color

    def addSunLight(self, rotation_euler, strength, shadow_soft_size = 0.05):
        x = rotation_euler[0] * 1.0 / 180.0 * np.pi 
        y = rotation_euler[1] * 1.0 / 180.0 * np.pi 
        z = rotation_euler[2] * 1.0 / 180.0 * np.pi 
        angle = (x,y,z)
        bpy.ops.object.light_add(type = 'SUN', rotation = angle)
        lamp = bpy.data.lights['Sun']
        lamp.use_nodes = True
        # lamp.shadow_soft_size = shadow_soft_size # this is for older blender 2.8
        lamp.angle = shadow_soft_size

        lamp.node_tree.nodes["Emission"].inputs['Strength'].default_value = strength
