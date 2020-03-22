import random
import json

class Color:
    def __init__(self, R, G, B, A=1.0):
        self.R = R
        self.G = G 
        self.B = B 
        self.A = A

    def RGBA(self):
        return (self.R, self.G,self.B,self.A)
    
    def RGB(self):
        return (self.R, self.G,self.B)

#https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html
class PrincipledBSDF:
    def __init__(self):
        self.setToDefaults()

    def setToDefaults(self):
        # Diffuse or metal surface color.
        self.BaseColor = Color(0.5, 0.1, 0.1)
        # Mix between diffuse and subsurface scattering. 
        self.Subsurface = 0.0    
        # Average distance that light scatters below the surface.
        self.SubsurfaceRadius = Color(1.0,0.2,0.1)
        # Subsurface scattering base color.
        self.SubsurfaceColor = Color(0.8, 0.8, 0.8)
        # Blends between a non-metallic and metallic material model. 
        self.Metallic = 0.0
        # Amount of dielectric specular reflection. 
        self.Specular = 0.5
        # Tints the facing specular reflection using the base color, while glancing reflection remains white.
        self.SpecularTint = 0.0
        # Specifies microfacet roughness of the surface for diffuse and specular reflection.
        self.Roughness = 0.5
        # Amount of anisotropy for specular reflection. 
        self.Anisotropic = 0.0
        # Rotates the direction of anisotropy, with 1.0 going full circle.
        self.AnisotropicRotation = 0.0
        # Amount of soft velvet like reflection near edges, for simulating materials such as cloth.
        self.Sheen = 0.0
        # Mix between white and using base color for sheen reflection.
        self.SheenTint = 0.5
        # Extra white specular layer on top of others.
        self.Clearcoat = 0.0
        # Roughness of clearcoat specular.
        self.ClearcoatRoughness = 0.003
        # Index of refraction for transmission.
        self.IOR = 1.45
        # Mix between fully opaque surface at zero and fully glass like transmission at one.
        self.Transmission = 0.0
        # With GGX distribution controls roughness used for transmitted light.
        self.TransmissionRoughness = 0.0
        # Light emission from the surface, like the Emission shader.
        self.Emission = Color(0.0,0.0,0.0)
        # Controls the transparency of the surface, with 1.0 fully opaque.
        self.Alpha = 1.0

    def setNode(self, bsdfNode):
        bsdfNode.inputs["Base Color"].default_value = self.BaseColor.RGBA()
        bsdfNode.inputs["Subsurface"].default_value = self.Subsurface
        bsdfNode.inputs["Subsurface Radius"].default_value = self.SubsurfaceRadius.RGB()
        bsdfNode.inputs["Subsurface Color"].default_value = self.SubsurfaceColor.RGBA()
        bsdfNode.inputs["Metallic"].default_value = self.Metallic
        bsdfNode.inputs["Specular"].default_value = self.Specular
        bsdfNode.inputs["Specular Tint"].default_value = self.SpecularTint
        bsdfNode.inputs["Roughness"].default_value = self.Roughness
        bsdfNode.inputs["Anisotropic"].default_value = self.Anisotropic
        bsdfNode.inputs["Anisotropic Rotation"].default_value = self.AnisotropicRotation
        bsdfNode.inputs["Sheen"].default_value = self.Sheen
        bsdfNode.inputs["Sheen Tint"].default_value = self.SheenTint
        bsdfNode.inputs["Clearcoat"].default_value = self.Clearcoat
        bsdfNode.inputs["Clearcoat Roughness"].default_value = self.ClearcoatRoughness
        bsdfNode.inputs["IOR"].default_value = self.IOR
        bsdfNode.inputs["Transmission"].default_value = self.Transmission
        bsdfNode.inputs["Transmission Roughness"].default_value = self.TransmissionRoughness
        bsdfNode.inputs["Emission"].default_value = self.Emission.RGBA()
        bsdfNode.inputs["Alpha"].default_value = self.Alpha

    def randomize(self):
        r = lambda: random.random()

        self.BaseColor = Color(r(), r(), r())
        self.Subsurface = r()   
        self.SubsurfaceRadius = Color(r(),r(),r())
        self.SubsurfaceColor = Color(r(), r(), r())
        self.Metallic = r()
        self.Specular = r()
        self.SpecularTint = r()
        self.Roughness = r()
        self.Anisotropic = r()
        self.AnisotropicRotation = r()
        self.Sheen = r()
        self.SheenTint = r()
        self.Clearcoat = r()
        self.ClearcoatRoughness = r()
        self.IOR = 1.45
        self.Transmission = r()
        self.TransmissionRoughness = r()
        self.Emission = Color(0.0,0.0,0.0)
        self.Alpha = 1.0

    def r(self):
        return random.random()

    def save(self, filename):
        ## TODO: write as a json config file? 
        materialConfig = {
            'Base Color': self.BaseColor.RGBA(),
            'Subsurface': self.Subsurface,
            'Subsurface Radius': self.SubsurfaceRadius.RGB(),
            'Subsurface Color': self.SubsurfaceColor.RGBA(),
            'Metallic': self.Metallic,
            'Specular': self.Specular,
            'Specular Tint': self.SpecularTint,
            'Roughness': self.Roughness,
            'Anisotropic': self.Anisotropic,
            'Anisotropic Rotation': self.AnisotropicRotation,
            'Sheen': self.Sheen,
            'Sheen Tint': self.SheenTint,
            'Clearcoat': self.Clearcoat,
            'Clearcout Roughness': self.ClearcoatRoughness,
            'IOR': self.IOR,
            'Transmission': self.Transmission,
            'Transmission Roughness': self.TransmissionRoughness,
            'Emission': self.Emission.RGBA(),
            'Alpha': self.Alpha
        }

        configJson = json.dumps(materialConfig)
        f = open(filename,"w")
        f.write(configJson)
        f.close()
    


