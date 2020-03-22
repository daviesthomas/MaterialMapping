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
	renderer = Render(256, 256, exposure=1, numSamples=64)
	
	scene = Scene()

	scene.setAmbientLight((0.2,0.2,0.2,1))
	scene.addSunLight((-15, -34, -155), 2, 0.1)

	geometry = Geometry('examples/blenderSphere.stl', rotation_euler=(0,0.0,90.0))
	#subdivide for smooth shading
	geometry.subdivide(2)

	camLocation = (1.9,2,2.2)
	lookAtLocation = (0,0,0.5)
	cam = Camera(location=camLocation, lookat=lookAtLocation, focalLength= 45).bpyCam()

	for i in range (200):
		bsdfMaterial = PrincipledBSDF()
		bsdfMaterial.randomize()

		geometry.setBSDF(bsdfMaterial)

		renderer.run('examples/renders/render-{}.png'.format(i), cam)
		bsdfMaterial.save('examples/renders/material-{}.json'.format(i))


main()