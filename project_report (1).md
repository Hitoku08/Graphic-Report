# Implementation and Optimization of Real-Time Ray Tracing Using NVIDIA OptiX

## Abstract

This report presents a comprehensive implementation of hardware-accelerated ray tracing using NVIDIA OptiX. Beginning with fundamental pipeline setup, we progressively develop a complete path tracer capable of accurately simulating global illumination. We demonstrate the implementation of core rendering techniques including ray-geometry intersections, texture mapping, shadow and transmission rays, temporal denoising, and multiple-bounce global illumination with Monte Carlo integration. Performance optimizations using hardware acceleration and variance reduction strategies are discussed. Experimental results show significant visual quality improvements through our implementation of physically-based rendering methods, demonstrating the capability of modern GPUs to achieve realistic lighting simulation in real-time applications.

## 1. Introduction

Ray tracing has traditionally been considered too computationally intensive for real-time applications. However, with dedicated ray tracing hardware in modern GPUs and acceleration frameworks like NVIDIA OptiX, real-time ray tracing has become feasible. This project explores the implementation of progressive ray tracing techniques using the OptiX framework, showcasing the capabilities of hardware-accelerated ray tracing for realistic rendering.

The OptiX framework provides a programmable ray tracing pipeline that leverages GPU acceleration for intersection testing and shading operations. It offers capabilities for building and traversing acceleration structures, handling ray-geometry intersections, and managing complex shading operations. Our implementation progressively builds upon this framework to achieve increasingly realistic rendering effects, culminating in a physically-based path tracer.

### 1.1 Project Objectives

The primary objectives of this project are:

1. Establish a framework for hardware-accelerated ray tracing using NVIDIA OptiX
2. Implement essential ray tracing components including ray generation, intersection, and shading
3. Support complex scene loading and texture mapping for realistic material representation
4. Implement advanced rendering techniques including shadows, transmission, and temporal denoising
5. Extend the system to physically-based path tracing with multiple bounces and Monte Carlo integration

## 2. Background and Related Work

### 2.1 Ray Tracing Fundamentals

Ray tracing simulates the physical behavior of light by tracing the path of light rays as they interact with objects in a scene. The technique involves generating rays from a viewer's perspective (primary rays), calculating intersections with scene geometry, and recursively generating secondary rays to simulate reflection, refraction, and shadow effects.

### 2.2 NVIDIA OptiX

OptiX is a programmable ray tracing framework developed by NVIDIA that provides a flexible pipeline for implementing ray tracing algorithms on GPU hardware. It offers capabilities for:

- Building and traversing acceleration structures for efficient ray-geometry intersection testing
- Managing the ray tracing pipeline through customizable programs for ray generation, intersection, closest hit, miss, and exception handling
- Hardware acceleration of ray traversal operations using dedicated RT cores on supported GPUs

### 2.3 Path Tracing and Global Illumination

Path tracing is an extension of ray tracing that aims to solve the rendering equation by Monte Carlo integration. It traces multiple light paths from the camera and accounts for all light transport paths, including direct illumination, indirect illumination, reflections, and refractions. This approach provides physically accurate global illumination but requires significant computational resources to reduce noise through sufficient sampling.

## 3. Basic OptiX Framework and Environment

Our implementation begins with establishing the OptiX ray tracing pipeline and environment setup. The essential components include:

- OptiX context initialization
- Acceleration structure creation
- Program binding for ray generation, intersection, and shading operations

The initial implementation focuses on creating a simple ray tracer capable of rendering basic geometric primitives with simple shading models. This foundation serves as the starting point for more advanced techniques implemented in subsequent stages.

[INSERT FIGURE: Basic OptiX pipeline diagram showing the relationship between ray generation, traversal, and shading programs]

## 4. Model Loading and Ray-Geometry Intersection

Building upon the basic framework, we implement model loading and ray-geometry intersection for rendering complex scenes. Key components include:

### 4.1 Model Loading

We implement model loading capabilities to support common mesh formats (OBJ/GLB). The loaded geometry is organized into a hierarchical structure for efficient intersection testing:

```
function LoadModel(filepath):
    model ← ParseModelFile(filepath)
    
    // Create geometry instances for each mesh
    for each mesh in model:
        vertices ← mesh.vertices
        indices ← mesh.indices
        
        // Create vertex and index buffers on device
        vertexBuffer ← CreateBuffer(vertices)
        indexBuffer ← CreateBuffer(indices)
        
        // Create geometry instance
        geometryInstance ← CreateGeometryInstance()
        geometryInstance.SetVertexBuffer(vertexBuffer)
        geometryInstance.SetIndexBuffer(indexBuffer)
        geometryInstance.SetIntersectionProgram(triangleIntersection)
        geometryInstance.SetClosestHitProgram(triangleClosestHit)
        
        geometryInstances.Add(geometryInstance)
    
    // Create geometry group and acceleration structure
    geometryGroup ← CreateGeometryGroup(geometryInstances)
    accelerationStructure ← BuildAccelerationStructure(geometryGroup)
    
    return accelerationStructure
```

### 4.2 Ray-Geometry Intersection

The ray-geometry intersection process involves:

1. Ray generation from the camera
2. Traversal of the acceleration structure to find potential intersections
3. Detailed intersection testing with primitives
4. Shading computation at intersection points

For triangle meshes, we implement an optimized intersection program using barycentric coordinates:

```
function IntersectTriangle(ray, vertices, indices, triIndex):
    // Get triangle vertices
    i0 ← indices[3 * triIndex]
    i1 ← indices[3 * triIndex + 1]
    i2 ← indices[3 * triIndex + 2]
    
    v0 ← vertices[i0]
    v1 ← vertices[i1]
    v2 ← vertices[i2]
    
    // Compute triangle edges
    e1 ← v1 - v0
    e2 ← v2 - v0
    
    // Calculate determinant
    pvec ← cross(ray.direction, e2)
    det ← dot(e1, pvec)
    
    // Check if ray is parallel to triangle
    if |det| < EPSILON:
        return false
    
    invDet ← 1.0 / det
    
    // Calculate barycentric coordinates
    tvec ← ray.origin - v0
    u ← dot(tvec, pvec) * invDet
    
    if u < 0.0 or u > 1.0:
        return false
    
    qvec ← cross(tvec, e1)
    v ← dot(ray.direction, qvec) * invDet
    
    if v < 0.0 or u + v > 1.0:
        return false
    
    // Compute intersection distance
    t ← dot(e2, qvec) * invDet
    
    if t > ray.tmin and t < ray.tmax:
        // Report intersection
        ReportIntersection(t, 0, u, v)
        return true
    
    return false
```

[INSERT FIGURE: Visualization of ray-triangle intersection showing barycentric coordinates]

## 5. Advanced Rendering Techniques

The core of our implementation focuses on advanced rendering techniques to improve visual quality and physical accuracy.

### 5.1 Complex Model Rendering and Material System

Our system supports loading and rendering complex models with diverse materials. Each material is represented by parameters including:

- Base color (albedo)
- Roughness and metalness for physically-based rendering (PBR)
- Emission properties for light sources
- Transmission properties for transparent materials

Materials are applied during the closest hit shader execution:

```
function ClosestHitShader(hitInfo):
    materialId ← GetMaterialId(hitInfo.primitiveIndex)
    material ← materials[materialId]
    
    // Get surface properties at hit point
    position ← ray.origin + ray.direction * hitInfo.t
    barycentric ← hitInfo.barycentric
    triangleIndex ← hitInfo.primitiveIndex
    
    // Interpolate attributes using barycentric coordinates
    normal ← InterpolateNormal(triangleIndex, barycentric)
    texCoord ← InterpolateTexCoord(triangleIndex, barycentric)
    
    // Texture lookups if available
    if material.hasAlbedoTexture:
        albedo ← SampleTexture(material.albedoTexture, texCoord)
    else:
        albedo ← material.albedo
    
    // Additional material properties
    roughness ← material.roughness
    metalness ← material.metalness
    emission ← material.emission
    
    // Perform shading
    return ShadeSurface(position, normal, albedo, roughness, metalness, emission)
```

### 5.2 Texture Mapping

Texture mapping is implemented to enhance visual detail without increasing geometric complexity. Our implementation supports:

- Diffuse/albedo texture maps for surface color
- Normal maps for detailed surface perturbation
- Roughness/metalness maps for physically-based material properties

Texture mapping is performed by:

1. Interpolating texture coordinates at intersection points using barycentric coordinates
2. Sampling texture data with appropriate filtering
3. Applying sampled values to the material shading model

### 5.3 Shadow and Transmission Rays

#### 5.3.1 Shadow Rays

Shadow rays determine visibility between intersection points and light sources:

```
function TraceShadowRay(hitPoint, lightPosition, normal):
    shadowRayOrigin ← hitPoint + normal * EPSILON  // Offset to avoid self-intersection
    shadowRayDirection ← normalize(lightPosition - hitPoint)
    distanceToLight ← length(lightPosition - hitPoint)
    
    shadowRay ← CreateRay(shadowRayOrigin, shadowRayDirection, 0.0, distanceToLight)
    shadowHit ← TraceRay(shadowRay, ANY_HIT)
    
    if shadowHit:
        return 0.0  // Point is in shadow
    else:
        return 1.0  // Point is visible to light
```

#### 5.3.2 Transmission Rays

For transparent materials, we implement transmission rays to simulate refraction:

```
function TraceTransmissionRay(hitPoint, normal, incident, ior):
    // Compute refraction direction using Snell's law
    cosTheta ← dot(-incident, normal)
    etaRatio ← cosTheta > 0.0 ? 1.0/ior : ior
    normalSign ← cosTheta > 0.0 ? 1.0 : -1.0
    cosTheta ← abs(cosTheta)
    
    sinTheta ← sqrt(1.0 - cosTheta * cosTheta)
    sinThetaPrime ← etaRatio * sinTheta
    
    // Check for total internal reflection
    if sinThetaPrime >= 1.0:
        // Total internal reflection
        reflectionDir ← reflect(incident, normal * normalSign)
        reflectionRay ← CreateRay(hitPoint + reflectionDir * EPSILON, reflectionDir)
        return TraceRay(reflectionRay)
    
    // Compute refraction direction
    cosThetaPrime ← sqrt(1.0 - sinThetaPrime * sinThetaPrime)
    refractionDir ← etaRatio * incident + (etaRatio * cosTheta - cosThetaPrime) * normal * normalSign
    
    // Create and trace refraction ray
    refractionRay ← CreateRay(hitPoint - normal * normalSign * EPSILON, refractionDir)
    return TraceRay(refractionRay)
```

[INSERT FIGURE: Comparison showing scene with and without shadows and transmission effects]

### 5.4 Temporal Denoising

Ray tracing with limited samples per pixel produces noisy results. We implement temporal denoising to accumulate samples across frames and reduce variance:

```
function TemporalDenoise(currentFrame, previousFrame, motionVectors):
    for each pixel (x, y):
        // Reproject previous pixel using motion vectors
        prevX ← x - motionVectors[x, y].x
        prevY ← y - motionVectors[x, y].y
        
        // Check if reprojection is valid
        if IsValidReprojection(prevX, prevY):
            previousColor ← SampleWithBilinearFilter(previousFrame, prevX, prevY)
            
            // Compute color difference and detect disocclusions
            colorDifference ← length(currentFrame[x, y] - previousColor)
            if colorDifference < threshold:
                // Blend current and previous frame
                alpha ← min(1.0, 1.0 / frameCount)
                result[x, y] ← lerp(previousColor, currentFrame[x, y], alpha)
            else:
                // Disocclusion detected, use current frame
                result[x, y] ← currentFrame[x, y]
        else:
            // Invalid reprojection, use current frame
            result[x, y] ← currentFrame[x, y]
    
    return result
```

The temporal denoising significantly improves image quality by reducing noise while preserving detail, particularly beneficial for scenes with limited sample budgets.

## 6. Path Tracing and Global Illumination

The final stage of our implementation extends the ray tracing system to full path tracing with global illumination.

### 6.1 The Rendering Equation

Path tracing aims to solve the rendering equation, which describes the equilibrium distribution of radiance in a scene:

$$L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{\Omega} f_r(x, \omega_i, \omega_o) L_i(x, \omega_i) (\omega_i \cdot n) d\omega_i$$

Where:
- $L_o(x, \omega_o)$ is the outgoing radiance from point $x$ in direction $\omega_o$
- $L_e(x, \omega_o)$ is the emitted radiance
- $f_r(x, \omega_i, \omega_o)$ is the bidirectional reflectance distribution function (BRDF)
- $L_i(x, \omega_i)$ is the incoming radiance from direction $\omega_i$
- $(\omega_i \cdot n)$ accounts for the orientation of the surface

### 6.2 Multiple-Bounce Path Tracing

We implement path tracing with multiple bounces to account for indirect illumination:

```
function TracePath(ray, maxDepth):
    if maxDepth <= 0:
        return vec3(0.0)  // Path terminated
    
    hitInfo ← TraceRay(ray)
    
    if not hitInfo.hit:
        return SampleEnvironment(ray.direction)  // Miss - return environment light
    
    // Get material properties at hit point
    material ← GetMaterial(hitInfo)
    position ← ray.origin + ray.direction * hitInfo.t
    normal ← hitInfo.normal
    
    // Handle emissive surfaces
    if material.isEmissive:
        return material.emission
    
    // Compute direct lighting contribution
    directLighting ← ComputeDirectLighting(position, normal, material)
    
    // Sample BRDF for indirect lighting direction
    wi ← SampleBRDF(material, normal, -ray.direction)
    pdf ← ComputePDF(material, normal, -ray.direction, wi)
    
    // Russian roulette termination
    terminationProbability ← max(0.05, 1.0 - max(material.albedo.r, max(material.albedo.g, material.albedo.b)))
    if random() < terminationProbability and maxDepth > 1:
        return directLighting
    
    // Recursive ray for indirect lighting
    nextRay ← CreateRay(position + normal * EPSILON, wi)
    indirectLighting ← TracePath(nextRay, maxDepth - 1)
    
    // Compute BRDF value
    brdf ← EvaluateBRDF(material, normal, -ray.direction, wi)
    
    // Compute contribution with proper weighting
    indirectContribution ← (brdf * indirectLighting * (wi.dot(normal))) / (pdf * (1.0 - terminationProbability))
    
    return directLighting + indirectContribution
```

### 6.3 Monte Carlo Integration for the Rendering Equation

To estimate the integral in the rendering equation, we use Monte Carlo integration with importance sampling:

```
function ComputeDirectLighting(position, normal, material):
    directLighting ← vec3(0.0)
    
    // Sample lights in the scene
    for each light in lights:
        // Sample a random point on the light
        lightSample ← SampleLight(light)
        lightDirection ← normalize(lightSample.position - position)
        
        // Check visibility
        visibility ← TraceShadowRay(position, lightSample.position, normal)
        
        if visibility > 0.0:
            // Compute lighting contribution
            distanceSquared ← lengthSquared(lightSample.position - position)
            nDotL ← max(0.0, dot(normal, lightDirection))
            
            // Evaluate BRDF
            brdf ← EvaluateBRDF(material, normal, viewDir, lightDirection)
            
            // Light contribution with correct weighting
            lightPDF ← lightSample.pdf
            contribution ← (lightSample.emission * brdf * nDotL * visibility) / (distanceSquared * lightPDF)
            
            directLighting += contribution
    
    return directLighting
```

### 6.4 BRDF Importance Sampling

To reduce variance in the Monte Carlo estimation, we implement importance sampling of the BRDF:

```
function SampleBRDF(material, normal, viewDir):
    if material.isMetallic:
        // Sample microfacet BRDF for metals
        alpha ← material.roughness * material.roughness
        
        // Sample microfacet normal using GGX distribution
        phi ← 2.0 * PI * random()
        cosTheta ← sqrt((1.0 - random()) / (1.0 + (alpha*alpha - 1.0) * random()))
        sinTheta ← sqrt(1.0 - cosTheta * cosTheta)
        
        // Convert to Cartesian coordinates in tangent space
        x ← sinTheta * cos(phi)
        y ← sinTheta * sin(phi)
        z ← cosTheta
        
        // Construct tangent space
        tangent, bitangent ← CreateOrthonormalBasis(normal)
        
        // Transform to world space
        h ← tangent * x + bitangent * y + normal * z
        
        // Reflect view direction around half vector
        return reflect(-viewDir, h)
    else:
        // Diffuse surface - cosine weighted hemisphere sampling
        phi ← 2.0 * PI * random()
        cosTheta ← sqrt(random())
        sinTheta ← sqrt(1.0 - cosTheta * cosTheta)
        
        // Convert to Cartesian coordinates in tangent space
        x ← sinTheta * cos(phi)
        y ← sinTheta * sin(phi)
        z ← cosTheta
        
        // Construct tangent space
        tangent, bitangent ← CreateOrthonormalBasis(normal)
        
        // Transform to world space
        return tangent * x + bitangent * y + normal * z
```

[INSERT FIGURE: Comparison of single-bounce direct illumination vs multi-bounce global illumination]

### 6.5 Variance Reduction Techniques

Several variance reduction techniques are implemented to improve convergence:

#### 6.5.1 Multiple Importance Sampling (MIS)

MIS combines multiple sampling strategies to reduce variance, particularly for scenes with both BRDF and light sampling:

```
function SampleWithMIS(position, normal, material, viewDir):
    result ← vec3(0.0)
    
    // BRDF sampling
    brdfDir ← SampleBRDF(material, normal, viewDir)
    brdfPDF ← ComputeBRDFPDF(material, normal, viewDir, brdfDir)
    
    // Trace ray in BRDF direction
    brdfRay ← CreateRay(position + normal * EPSILON, brdfDir)
    brdfHit ← TraceRay(brdfRay)
    
    // Check if we hit a light
    if brdfHit.hit and IsEmissive(brdfHit):
        // Get light properties
        lightEmission ← GetEmission(brdfHit)
        lightPDF ← ComputeLightPDF(brdfHit)
        
        // Evaluate BRDF
        brdf ← EvaluateBRDF(material, normal, viewDir, brdfDir)
        
        // MIS weight
        weight ← brdfPDF / (brdfPDF + lightPDF)
        
        // Add contribution
        result += weight * lightEmission * brdf * dot(normal, brdfDir) / brdfPDF
    
    // Light sampling
    lightSample ← SampleLight()
    lightDir ← normalize(lightSample.position - position)
    
    // Check if light is visible and above horizon
    if dot(normal, lightDir) > 0.0:
        shadowRay ← CreateRay(position + normal * EPSILON, lightDir)
        visible ← !TraceShadowRay(shadowRay)
        
        if visible:
            // Evaluate BRDF
            brdf ← EvaluateBRDF(material, normal, viewDir, lightDir)
            
            // Compute PDFs for MIS
            lightPDF ← lightSample.pdf
            brdfPDF ← ComputeBRDFPDF(material, normal, viewDir, lightDir)
            
            // MIS weight
            weight ← lightPDF / (brdfPDF + lightPDF)
            
            // Add contribution
            distance ← length(lightSample.position - position)
            attenuation ← 1.0 / (distance * distance)
            result += weight * lightSample.emission * brdf * dot(normal, lightDir) * attenuation / lightPDF
    
    return result
```

#### 6.5.2 Stratified Sampling

We implement stratified sampling to improve distribution of samples:

```
function GenerateStratifiedSamples(width, height, samplesPerPixel):
    sqrtSamples ← ceil(sqrt(samplesPerPixel))
    
    for y in [0, height):
        for x in [0, width):
            for i in [0, sqrtSamples):
                for j in [0, sqrtSamples):
                    // Stratified sample position within pixel
                    strataX ← (i + random()) / sqrtSamples
                    strataY ← (j + random()) / sqrtSamples
                    
                    // Generate ray for this sample
                    u ← (x + strataX) / width
                    v ← (y + strataY) / height
                    
                    ray ← GenerateCameraRay(u, v)
                    color ← TracePath(ray, maxDepth)
                    
                    // Accumulate result
                    framebuffer[x, y] += color / samplesPerPixel
```

### 6.6 Progressive Refinement

To provide interactive feedback during rendering, we implement progressive refinement:

```
function ProgressiveRender(scene, camera):
    samplesPerPixel ← 1
    accumulatedSamples ← 0
    
    while not terminated:
        for each pixel (x, y):
            // Generate stratified samples for this iteration
            for i in [0, samplesPerPixel):
                u ← (x + random()) / width
                v ← (y + random()) / height
                
                ray ← GenerateCameraRay(camera, u, v)
                color ← TracePath(ray, maxDepth)
                
                // Accumulate result
                accumulationBuffer[x, y] += color
            
        // Update display
        accumulatedSamples += samplesPerPixel
        for each pixel (x, y):
            framebuffer[x, y] ← accumulationBuffer[x, y] / accumulatedSamples
        
        // Optional: apply temporal denoising
        if enableDenoising:
            framebuffer ← TemporalDenoise(framebuffer)
        
        DisplayFrame(framebuffer)
        
        // Adaptive sample count increase
        samplesPerPixel ← min(maxSamplesPerIteration, samplesPerPixel * 2)
```

[INSERT FIGURE: Series of images showing progressive refinement of path tracing over increasing sample counts]

## 7. Results and Performance Analysis

Our implementation achieves realistic rendering results through physically-based path tracing with multiple bounces. The key visual improvements include:

- Accurate soft shadows from area lights
- Color bleeding from indirect illumination
- Realistic reflections and refractions
- Global illumination with light transport throughout the scene

### 7.1 Visual Quality Analysis

[INSERT FIGURE: Comparison between rasterization, single-bounce ray tracing, and full path tracing of the same scene]

The path tracing implementation significantly improves visual quality compared to traditional rasterization or single-bounce ray tracing. Key improvements include:

- Physically accurate light transport and energy conservation
- Soft shadows with proper penumbra regions
- Indirect illumination capturing color bleeding between surfaces
- Accurate reflections and refractions with Fresnel effects

### 7.2 Performance Analysis

Performance measurements were conducted on several GPU configurations to evaluate the impact of hardware ray tracing acceleration:

[INSERT TABLE: Performance comparison across different GPUs and rendering techniques]

The analysis shows:
- Hardware acceleration provides significant performance benefits, particularly for complex scenes
- Temporal denoising effectively reduces the required sample count for acceptable image quality
- Multiple importance sampling improves convergence rates for scenes with complex lighting

## 8. Future Work

Several potential areas for future improvement include:

1. **Advanced Material Models**: Implementation of more sophisticated material models such as subsurface scattering, participating media, and volumetric rendering.

2. **Bidirectional Path Tracing**: Extension to bidirectional path tracing to better handle difficult light transport paths.

3. **Adaptive Sampling**: Implementation of adaptive sampling techniques to focus computational resources on high-variance regions.

4. **Machine Learning Denoising**: Integration of neural network-based denoising techniques to further improve image quality with limited samples.

5. **Out-of-Core Rendering**: Support for rendering scenes that exceed GPU memory through streaming techniques.

## 9. Conclusion

This project demonstrates the implementation of a physically-based renderer using NVIDIA OptiX for hardware-accelerated ray tracing. Starting from basic ray-geometry intersection handling, we progressively implemented advanced rendering techniques including texture mapping, shadows, transmission effects, temporal denoising, and full path tracing with global illumination.

The results showcase the capability of modern GPUs with dedicated ray tracing hardware to achieve realistic rendering with physically accurate light transport. The combination of hardware acceleration, variance reduction techniques, and temporal accumulation enables interactive path tracing performance with progressive refinement to high-quality results.

The implementation provides a solid foundation for further research and development in real-time ray tracing applications, with potential applications in areas such as architectural visualization, product design, virtual reality, and interactive entertainment.

## References

1. J. T. Kajiya, "The rendering equation," SIGGRAPH '86, pp. 143-150, 1986.
2. S. G. Parker et al., "OptiX: A general purpose ray tracing engine," ACM Trans. Graph., vol. 29, no. 4, 2010.
3. T. Whitted, "An improved illumination model for shaded display," Commun. ACM, vol. 23, no. 6, pp. 343-349, 1980.
4. E. Veach and L. J. Guibas, "Optimally combining sampling techniques for Monte Carlo rendering," SIGGRAPH '95, pp. 419-428, 1995.
5. E. Veach, "Robust Monte Carlo methods for light transport simulation," Ph.D. dissertation, Stanford University, 1997.
6. M. Pharr, W. Jakob, and G. Humphreys, "Physically Based Rendering: From Theory to Implementation," 3rd ed. Morgan Kaufmann, 2016.
7. C. Crassin et al., "Interactive indirect illumination using voxel cone tracing," Computer Graphics Forum, vol. 30, no. 7, pp. 1921-1930, 2011.
8. T. Schied et al., "Spatiotemporal variance-guided filtering: Real-time reconstruction for path-traced global illumination," High Performance Graphics, 2017.
