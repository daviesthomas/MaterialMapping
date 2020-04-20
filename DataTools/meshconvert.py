# easy tool too have on hand for quick mesh conversion

import trimesh
import argparse
import os

def asMesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputFolder")
    parser.add_argument("--normalize", default=1)
    args = parser.parse_args()

    meshToConvert = [os.path.join(args.inputFolder,f) for f in os.listdir(args.inputFolder) if '.obj' in f]

    print(meshToConvert)
    for fn in meshToConvert:
        mesh = asMesh(trimesh.load_mesh(fn))
        stlPath = fn.replace('.obj', '.stl')
        print(stlPath)
        mesh.export(stlPath)

main()