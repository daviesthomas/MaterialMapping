import bpy
import numpy as np
import bmesh
import mathutils
import argparse

import os 
import sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

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
	parser.add_argument('-j', '--joinMaterial', default=False)
	parser.add_argument('-a', '--antiAliasing',default=2)

	args = parser.parse_args()

	materialFiles = [os.path.join(args.materialFolder, p) for p in os.listdir(args.materialFolder) if '.json' in p]
	
	if (os.path.isdir(args.meshFolder)):
		geometryFiles = [os.path.join(args.meshFolder, p) for p in os.listdir(args.meshFolder)]
	else:
		geometryFiles = [args.meshFolder]

	# setup constants
	renderer = Render(args.imageWidth*int(args.antiAliasing),args.imageHeight*int(args.antiAliasing), exposure=1, numSamples=args.numSamples)
	scene = Scene()
	scene.setAmbientLight((0.2,0.2,0.2,1))
	scene.addSunLight((-15, -34, -155), 2, 0.1)
	
	camLocation = (1.4,1.5,1.6)
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

			if (args.antiAliasing > 1):
				# open image, resize, save
				image = Image.open(renderPath)
				aaImage = image.resize((args.imageWidth, args.imageHeight))
				aaImage.save(renderPath)

			if (args.joinMaterial):
				#concatenate the two images
				jointImagePath = 'examples/pairs/{}_{}.png'.format(
					os.path.basename(os.path.splitext(geometryFile)[0]),
					materialFile.split('-')[-1].split('.')[0])

				newW = int(args.imageWidth/2)
				newH = int(args.imageHeight/2)

				materialImage = Image.open(materialFile.replace('.json', '.png'))
				materialImage = materialImage.resize((newW, newH))	

				renderImage = Image.open(renderPath)
				renderImage = renderImage.resize((newW,newH))

				print((int(newW/2), int(newH/2)), (2*int(newW/2) + newW, int(newH/2)))
				jointImage = Image.new('RGBA', (args.imageWidth, args.imageHeight))
				jointImage.paste(materialImage, (0, int(newH/2)))
				jointImage.paste(renderImage, (newW, int(newH/2)))

				jointImage.save(jointImagePath)

				

				

				
				

		geometry.delete()

main()