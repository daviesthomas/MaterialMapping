import bpy
import numpy as np
import bmesh
import mathutils
import argparse

import os 
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from blender.geometry import Geometry
from blender.material import PrincipledBSDF, Color
from blender.camera import Camera
from blender.scene import Scene
from blender.render import Render
from blender.argparser import ArgumentParserForBlender

def main():
	# given a folder of mesh, and folder of materials this tool will render each mesh with each material.
	parser = ArgumentParserForBlender()
	parser.add_argument('-i','--meshFolder',default='examples/geometries/')
	parser.add_argument('-m','--materialFolder', default='examples/materials/')
	parser.add_argument('-o','--renderFolder', default='examples/renders/')
	parser.add_argument('-H','--imageHeight', default=256)
	parser.add_argument('-W','--imageWidth', default=256)
	parser.add_argument('-s','--numSamples', default=128)	#rendering quality essentially.

	args = parser.parse_args()

	materialFiles = [os.path.join(args.materialFolder, p) for p in os.listdir(args.materialFolder) if '.json' in p]
	geometryFiles = [os.path.join(args.meshFolder, p) for p in os.listdir(args.meshFolder)]

	# setup constants
	renderer = Render(args.imageWidth,args.imageHeight, exposure=1, numSamples=args.numSamples)
	scene = Scene()
	scene.setAmbientLight((0.2,0.2,0.2,1))
	scene.addSunLight((-15, -34, -155), 2, 0.1)
	
	camLocation = (1.9,2,2.2)
	lookAtLocation = (0,0,0.5)
	cam = Camera(location=camLocation, lookat=lookAtLocation, focalLength= 45).bpyCam()

	for geometryFile in geometryFiles:
		geometry = Geometry(geometryFile, rotation_euler=(0,0,90))
		geometry.subdivide(2)	# not sure we always need this... but easier to not check...

		for materialFile in materialFiles:
			bsdfMaterial = PrincipledBSDF(materialFile)

			geometry.setBSDF(bsdfMaterial)
			renderPath = 'examples/renders/{}_{}.png'.format(
				os.path.basename(os.path.splitext(geometryFile)[0]),
				materialFile.split('-')[-1].split('.')[0])
			
			renderer.run(renderPath, cam)

		geometry.delete()

main()