import mathutils
import bpy

class Camera():
    def __init__(self, location=(2.0,2.0,2.0), lookat=(0.0,0.0,0.0), focalLength=45):
        bpy.ops.object.camera_add(location=location)
        self.camera = bpy.context.object
        self.camera.data.lens = focalLength
        self.__lookAt(lookat)

    def __lookAt(self, point):
        direction = mathutils.Vector(point) - self.camera.location
        rotQuat = direction.to_track_quat('-Z', 'Y')
        #rotate camera to look at point
        self.camera.rotation_euler = rotQuat.to_euler()

    def bpyCam(self):
        return self.camera

    
