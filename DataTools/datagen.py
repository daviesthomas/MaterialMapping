import bpy
import numpy as np
import bmesh
import mathutils

import os 
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from blender.geometry import Geometry
from blender.material import PrincipledBSDF, Color
from blender.camera import Camera
from blender.scene import Scene
from blender.render import Render

def main():
	renderer = Render(1024, 1024, exposure=1)
	
	scene = Scene()

	scene.setAmbientLight((0.2,0.2,0.2,1))
	scene.addSunLight((-15, -34, -155), 2, 0.1)

	geometry = Geometry('examples/gear_knee.stl')
	#subdivide for smooth shading
	geometry.subdivide(2)
	
	bsdfMaterial = PrincipledBSDF()
	geometry.setBSDF(bsdfMaterial)

	camLocation = (1.9,2,2.2)
	lookAtLocation = (0,0,0.5)
	cam = Camera(location=camLocation, lookat=lookAtLocation, focalLength= 45).bpyCam()

	renderer.run('test.png', cam)

main()