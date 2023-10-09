
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>

const int X_RAYAMOUNT = 3;
const int Y_RAYAMOUNT = 3;

const int PLANEAMOUNT = 1;
const int THREADSPERBLOCK = 64;

__constant__ int MAXRAYREFLECTION = 10;
__constant__ float RAYABSORPTIONRATIO = 0.2;
__constant__ float RAYREFLECTIONRATIO = 0.1;

struct vec3
{
    float x;
    float y;
    float z;

    bool operator==(struct vec3& other_vec3)
    {
        return x == other_vec3.x && y == other_vec3.y && z == other_vec3.z;
    }

    struct vec3 operator-(struct vec3& other_vec3)
    {
        return { x - other_vec3.x, y - other_vec3.y, z - other_vec3.z };
    }

    struct vec3 operator+(struct vec3& other_vec3)
    {
        return { x + other_vec3.x, y + other_vec3.y, z + other_vec3.z };
    }

    struct vec3 operator*(struct vec3& other_vec3)
    {
        return { x * other_vec3.x, y * other_vec3.y, z * other_vec3.z };
    }

    struct vec3 operator*(float scalar) const 
    {
        return { x* scalar, y* scalar, z* scalar};
    }
};

struct ray
{
    struct vec3 origin;
    struct vec3 direction;
    float energy;
};

struct plane
{
    struct vec3 pa;
    struct vec3 pb;
    struct vec3 pc;
    struct vec3 pd;
    struct vec3 center;
    struct vec3 normal;
};

#pragma region DeviceFunctions

#pragma region vec3Operations

__device__ struct vec3 vec3Plus(struct vec3 first, struct vec3 second) 
{
    return { first.x + second.x, first.y + second.y, first.z + second.z };
}

__device__ struct vec3 vec3Minus(struct vec3 first, struct vec3 second)
{
    return { first.x - second.x, first.y - second.y, first.z - second.z };
}

__device__ struct vec3 vec3Multi(struct vec3 first, struct vec3 second)
{
    return { first.x * second.x, first.y * second.y, first.z * second.z };
}

__device__ struct vec3 vec3Multi(struct vec3 first, float multiplier)
{
    return { first.x * multiplier, first.y * multiplier, first.z * multiplier };
}

#pragma endregion;

__device__ struct vec3 d_planeGet(struct plane plane_toget, int index)
{
    switch (index)
    {
    case 0:
        return plane_toget.pa;
        break;

    case 1:
        return plane_toget.pb;
        break;

    case 2:
        return plane_toget.pc;
        break;

    default:
        return plane_toget.pd;
    }
}

// Função para calcular o produto vetorial entre dois vetores
__device__ vec3 d_cross(struct vec3 a, struct vec3 b)
{
    vec3 result;

    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;

    return result;
}

// Função para calcular o produto escalar (dot product) entre dois vetores
__device__ float d_dot(struct vec3 a, struct vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Função para normalizar um vetor vec3
__device__ struct vec3 d_normalize(struct vec3 v) {
    float len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

    // Verificar se o comprimento é não nulo para evitar divisão por zero
    if (len != 0.0f) {
        v.x /= len;
        v.y /= len;
        v.z /= len;
    }

    return v;
}

// Função para verificar se um ponto está dentro de um triângulo
__device__ bool isPointInsideTriangle(struct vec3 point, struct vec3 t1, struct vec3 t2, struct vec3 t3) {
    // Calcular vetores dos vértices do triângulo para o ponto de interseção
    vec3 v0 = vec3Minus(t2, t1);
    vec3 v1 = vec3Minus(t3, t1);
    vec3 v2 = vec3Minus(point, t1);

    // Calcular produtos escalares
    float dot00 = d_dot(v0, v0);
    float dot01 = d_dot(v0, v1);
    float dot02 = d_dot(v0, v2);
    float dot11 = d_dot(v1, v1);
    float dot12 = d_dot(v1, v2);

    // Calcular coordenadas baricêntricas
    float invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    // Verificar se as coordenadas baricêntricas estão dentro do intervalo [0, 1]
    return (u >= 0) && (v >= 0) && (u + v <= 1);
}

// calculate the energy of the ray on the reflected panels along the way, returns 0 if none is hitted
__device__ float calculateRayEnergy(struct plane* d_planes, struct ray ray, int current_reflection, int reflected_plane_index)
{
    if (current_reflection > MAXRAYREFLECTION)
    {
        return 0;
    }

    float smallestDistanceSquared = INFINITY;
    float distanceSquared;

    int closest_plane_index = -1;

    struct vec3 normal;
    struct vec3 closestPlaneNormal;
    struct vec3 triangle[3];

    float intersectionDistance;
    struct vec3 intersectionPoint;
    struct vec3 closestIntersectionPoint;

    struct vec3 distanceVector;

    /*
    printf("ray origin: (%.1f, %.1f, %.1f)\n", ray.origin.x, ray.origin.y, ray.origin.z);
    printf("ray direction: (%.1f, %.1f, %.1f)\n", ray.direction.x, ray.direction.y, ray.direction.z);
    */

    for (int i = 0; i < PLANEAMOUNT; i++)
    {
        if (i == reflected_plane_index)
        {
            continue; // Ignore if plane was the reflected before
        }

        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                triangle[k] = d_planeGet(d_planes[i], (k + i * 2) % 4);
            }
        }

        normal = d_cross(vec3Minus(triangle[1], triangle[0]), vec3Minus(triangle[2], triangle[0]));

        // Cálculo do ponto de interseção entre o raio e o plano do triângulo
        intersectionDistance = d_dot(vec3Minus(triangle[1], ray.origin), normal) / d_dot(ray.direction, normal);
        intersectionPoint = vec3Plus(ray.origin, vec3Multi(ray.direction, intersectionDistance));

        // Verificar se o ponto de interseção está dentro do triângulo
        if (isPointInsideTriangle(intersectionPoint, triangle[0], triangle[1], triangle[2]))
        {
            distanceVector = vec3Minus(ray.origin, intersectionPoint);
            distanceVector = vec3Multi(distanceVector, distanceVector);
            distanceSquared = distanceVector.x + distanceVector.y + distanceVector.z;

            if (distanceSquared < smallestDistanceSquared) 
            {
                smallestDistanceSquared = distanceSquared;
                closest_plane_index = i;

                closestPlaneNormal = normal;
                closestIntersectionPoint = intersectionPoint;
            }
        }
    }

    if (closest_plane_index != -1)
    {
        vec3 reflectedDirection = vec3Minus(closestIntersectionPoint, vec3Multi(closestPlaneNormal, 2.0 * d_dot(ray.direction, closestPlaneNormal)));

        struct ray reflectionRay;

        reflectionRay.origin = closestIntersectionPoint;
        reflectionRay.direction = reflectedDirection;

        // Calculating the absorved and the reflected energy
        float reflected_energy = ray.energy * RAYREFLECTIONRATIO;
        float absorved_energy = (ray.energy - reflected_energy) * RAYABSORPTIONRATIO;
        reflectionRay.energy = reflected_energy;

        // Calling for the reflected ray
        return absorved_energy + calculateRayEnergy(d_planes, reflectionRay, ++current_reflection, closest_plane_index);
    }

    return 0.0;
}

#pragma endregion

#pragma region GlobalFunctions

__global__ void receiveRaysAndPlanes(struct ray* d_rays, struct plane* d_planes, float* d_total_energy)
{
    int ray_index = threadIdx.x + blockIdx.x * blockDim.x;

    if (ray_index >= X_RAYAMOUNT * Y_RAYAMOUNT)
    {
        return;
    }

    if (d_rays[ray_index].direction.x == 0.0 && d_rays[ray_index].direction.y == 0.0 && d_rays[ray_index].direction.z == 0.0)
    {
        return;
    }

    float absorved_energy = calculateRayEnergy(d_planes, d_rays[ray_index], 0, -1);
    printf("%.2f\n", absorved_energy);

    atomicAdd(d_total_energy, absorved_energy);
}

#pragma endregion

#pragma region HostFuctions

#pragma region GetFunctions

__host__ float h_vec3Get(struct vec3 vec3_toget, int index)
{
    switch (index)
    {
    case 0:
        return vec3_toget.x;
        break;

    case 1:
        return vec3_toget.y;
        break;

    default:
        return vec3_toget.z;
    }
}

__host__ void h_vec3Set(struct vec3* vec3_toset, int index, float value)
{
    switch (index)
    {
    case 0:
        vec3_toset->x = value;
        break;

    case 1:
        vec3_toset->y = value;
        break;

    default:
        vec3_toset->z = value;
    }
}

__host__ struct vec3 h_planeGet(struct plane plane_toget, int index)
{
    switch (index)
    {
    case 0:
        return plane_toget.pa;
        break;

    case 1:
        return plane_toget.pb;
        break;

    case 2:
        return plane_toget.pc;
        break;

    default:
        return plane_toget.pd;
    }
}

#pragma endregion

// Função para normalizar um vetor vec3
__host__ struct vec3 h_normalize(struct vec3 v) {
    float len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

    // Verificar se o comprimento é não nulo para evitar divisão por zero
    if (len != 0.0f) {
        v.x /= len;
        v.y /= len;
        v.z /= len;
    }

    return v;
}

__host__ float rand_float(float min, float max)
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float range = max - min;
    return (random * range) + min;
}

__host__ void findPerpendicularVectors(vec3 direction, vec3* directionHeight, vec3* directionWidth)
{
    // Escolher um vetor de referência que não seja colinear com a normal
    vec3 referenceVector;

    if (direction.x != 0 || direction.y != 0) {
        referenceVector.x = 0;
        referenceVector.y = 0;
        referenceVector.z = 1;
    }
    else {
        referenceVector.x = 1;
        referenceVector.y = 0;
        referenceVector.z = 0;
    }

    // Calcular o primeiro vetor perpendicular usando o produto vetorial
    directionHeight->x = direction.y * referenceVector.z - direction.z * referenceVector.y;
    directionHeight->y = direction.z * referenceVector.x - direction.x * referenceVector.z;
    directionHeight->z = direction.x * referenceVector.y - direction.y * referenceVector.x;

    // Calcular o segundo vetor perpendicular usando o produto vetorial novamente
    directionWidth->x = direction.y * directionHeight->z - direction.z * directionHeight->y;
    directionWidth->y = direction.z * directionHeight->x - direction.x * directionHeight->z;
    directionWidth->z = direction.x * directionHeight->y - direction.y * directionHeight->x;
}

__host__ void CreateSetsOfRays(struct ray* ray_array, struct vec3 direction, struct vec3 origin, int collums, int rows, float distance, float total_energy)
{
    struct vec3 directionY;
    struct vec3 directionX;

    direction = h_normalize(direction);
    findPerpendicularVectors(direction, &directionY, &directionX);

    /*
    printf("RayDirection -> x: %.2f | y: %.2f | z: %.2f\n", direction->x, direction->y, direction->z);
    printf("RayY -> x: %.2f | y: %.2f | z: %.2f\n", directionY.x, directionY.y, directionY.z);
    printf("RayX -> x: %.2f | y: %.2f | z: %.2f\n\n", directionX.x, directionX.y, directionX.z);
    */

    float half_height = distance / 2;
    float half_width = distance / 2;

    vec3 current_point;
    int total_rays = 0;

    for (int i = 0; i < rows; i++)
    {
        current_point = origin + (directionY * half_height * (rows - 1 - (i * 2)) + directionX * half_width * (collums - 1) * -1);

        for (int j = 0; j < collums; j++)
        {
            //printf("\ncur point -> x: %.2f | y: %.2f | z: %.2f\n", current_point.x, current_point.y, current_point.z);

            ray_array[total_rays].origin = current_point;
            ray_array[total_rays].direction = direction;

            current_point = current_point + directionX * distance;
            total_rays++;
        }
    }

    float energy_per_ray = total_energy / total_rays;

    for (int i = 0; i < total_rays; i++)
    {
        ray_array[i].energy = energy_per_ray;
    }
}

#pragma region host vector triangle intersection functions

// Função para calcular o produto vetorial entre dois vetores
__host__ vec3 h_cross(struct vec3 a, struct vec3 b)
{
    vec3 result;

    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;

    return result;
}

// Função para calcular o produto escalar (dot product) entre dois vetores
__host__ float h_dot(struct vec3 a, struct vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Função para verificar interseção entre vetor e triângulo
__host__ bool checkIntersectionVectorTriangle(struct vec3 v1, struct vec3 v2, struct vec3 t1, struct vec3 t2, struct vec3 t3, struct vec3 normal )
{
    // Calcula a direção do vetor
    vec3 direction = v2 - v1;

    // Calcula o ponto de interseção entre o vetor e o plano do triângulo
    float planeConstant = -normal.x * t1.x - normal.y * t1.y - normal.z * t1.z;
    float intersectionParameter = -(normal.x * v1.x + normal.y * v1.y + normal.z * v1.z + planeConstant) / (normal.x * direction.x + normal.y * direction.y + normal.z * direction.z);

    // Verifica se o ponto de interseção está dentro dos limites do triângulo
    if (intersectionParameter >= 0 && intersectionParameter <= 1)
    {
        vec3 intersectionPoint = v1 + direction * intersectionParameter;

        // Verifique se o ponto de interseção está dentro do triângulo usando coordenadas barycentricas
        vec3 edge1 = t2 - t1;
        vec3 edge2 = t3 - t1;

        vec3 crossVec = h_cross(direction, edge2);
        float a = h_dot(edge1, crossVec);

        if (a != 0)
        {
            float u = h_dot(edge1, h_cross(v1 - t1, crossVec)) / a;
            float v = h_dot(direction, h_cross(v1 - t1, crossVec)) / a;

            if (u >= 0 && v >= 0 && u + v <= 1)
            {
                return true;
            }
        }
    }

    return false;
}

#pragma endregion

#pragma region planeCreation

//Divide o primeiro plano em vetores e o segundo plano em triângulos, verificando-os individualmente
__host__ bool checkIfPlanesCollide(struct plane plane1, struct plane plane2)
{
    if (plane1.center == plane2.center)
    {
        return true;
    }

    struct vec3 v1;
    struct vec3 v2;

    struct vec3 t1 = h_planeGet(plane2, 0);
    struct vec3 t2 = h_planeGet(plane2, 1);
    struct vec3 t3 = h_planeGet(plane2, 2);
    struct vec3 t4 = h_planeGet(plane2, 3);

    for (int i = 0; i < 3; i++)
    {
        v1 = h_planeGet(plane1, i);
        v2 = h_planeGet(plane1, i + 1);

        if (checkIntersectionVectorTriangle(v1, v2, t1, t2, t3, plane2.normal) || checkIntersectionVectorTriangle(v1, v2, t3, t4, t1, plane2.normal))
        {
            return true;
        }
    }

    return false; // Não há interseção
}

__host__ bool checkIfPlaneIsInBounds(struct plane plane_to_check, struct vec3 positive_limit, struct vec3 negative_limit)
{
    float negative_bound;
    float positive_bound;

    float plane_vertex_dim_value;

    for (int i = 0; i < 3; i++)
    {
        negative_bound = h_vec3Get(negative_limit, i);
        positive_bound = h_vec3Get(positive_limit, i);

        for (int j = 0; j < 4; j++)
        {
            plane_vertex_dim_value = h_vec3Get(h_planeGet(plane_to_check, j), i);

            if (plane_vertex_dim_value < negative_bound || plane_vertex_dim_value > positive_bound)
            {
                return false;
            }
        }
    }

    return true;
}

__host__ bool checkIfPlaneIsValid(struct plane plane_to_check, struct plane* plane_array, int plane_amount, struct vec3 positive_limit, struct vec3 negative_limit)
{
    for (int i = 0; i < plane_amount; i++)
    {
        if (checkIfPlanesCollide(plane_array[i], plane_to_check) || !checkIfPlaneIsInBounds(plane_to_check, positive_limit, negative_limit))
        {
            return false;
        }
    }

    return true;
}

__host__ struct plane CreatePlane(struct vec3 origin, struct vec3 direction, float plane_width, float plane_height)
{
    struct vec3 directionY;
    struct vec3 directionX;

    findPerpendicularVectors(direction, &directionY, &directionX);

    struct plane new_plane;

    new_plane.pa = origin + (directionY * (plane_height / 2)) + (directionX * (plane_width / 2));
    new_plane.pb = origin + (directionY * (plane_height / 2)) - (directionX * (plane_width / 2));
    new_plane.pc = origin - (directionY * (plane_height / 2)) - (directionX * (plane_width / 2));
    new_plane.pd = origin - (directionY * (plane_height / 2)) + (directionX * (plane_width / 2));

    new_plane.center = origin;
    new_plane.normal = direction;

    return new_plane;
}

// ---------------------------
// ----------- old -----------
// ---------------------------
__host__ void CreatePlaneOld(struct plane* plane_array, int plane_inx, struct vec3 origin, struct vec3 direction, float plane_width, float plane_height)
{
    struct vec3 directionY;
    struct vec3 directionX;

    direction = h_normalize(direction);

    findPerpendicularVectors(direction, &directionY, &directionX);

    printf("\nPlane: orientation -> x: %.1f | y: %.1f | z: %.1f\n", direction.x, direction.y, direction.z);
    printf("\nPlane: direction y -> x: %.1f | y: %.1f | z: %.1f\n", directionY.x, directionY.y, directionY.z);
    printf("\nPlane: direction x -> x: %.1f | y: %.1f | z: %.1f\n", directionX.x, directionX.y, directionX.z);

    struct plane new_plane;

    new_plane.pa = origin + (directionY * (plane_height / 2)) + (directionX * (plane_width / 2));
    new_plane.pb = origin + (directionY * (plane_height / 2)) - (directionX * (plane_width / 2));
    new_plane.pc = origin - (directionY * (plane_height / 2)) - (directionX * (plane_width / 2));
    new_plane.pd = origin - (directionY * (plane_height / 2)) + (directionX * (plane_width / 2));

    new_plane.center = origin;
    new_plane.normal = direction;

    plane_array[plane_inx] = new_plane;
}

__host__ void CreateSetsOfPlanes(struct plane* plane_array, int plane_amount, struct vec3 positive_limit, struct vec3 negative_limit, float plane_width, float plane_height)
{
    struct vec3 direction;
    struct vec3 point;

    int rejection_limit = 100;

    for (int i = 0; i < plane_amount; i++)
    {
        direction = h_normalize(vec3{ rand_float(0, 1), rand_float(0, 1), rand_float(0, 1) });
        point = { rand_float(negative_limit.x, positive_limit.x), rand_float(negative_limit.y, positive_limit.y), rand_float(negative_limit.z, positive_limit.z) };

        struct plane new_plane = CreatePlane(point, direction, plane_width, plane_height);
        new_plane.normal = direction;
        new_plane.center = point;

        printf("\ndirection -> x: %.2f | y: %.2f | z: %.2f\n", i, direction.x, direction.y, direction.z);

        if (!checkIfPlaneIsValid(new_plane, plane_array, i, positive_limit, negative_limit))
        {
            rejection_limit--;

            if (rejection_limit == 0)
            {
                break;
            }

            i--;
            continue;
        }

        plane_array[i] = new_plane;
    }
}

#pragma endregion

__host__ int ceil(int total, int threads)
{
    return (total - 1) / threads + 1;
}

#pragma endregion

int main()
{
    srand(static_cast<unsigned int>(time(nullptr)));

    struct plane planes[PLANEAMOUNT];
    struct plane* d_planes;
    int planes_size = sizeof(struct plane) * PLANEAMOUNT;

    struct ray rays[X_RAYAMOUNT * Y_RAYAMOUNT];
    struct ray* d_rays;
    int rays_size = sizeof(struct ray) * X_RAYAMOUNT * Y_RAYAMOUNT;

    float total_energy = 0.0;
    float* d_total_energy;

    CreateSetsOfRays(rays, { 0,0,1 }, { 0,0,0 }, X_RAYAMOUNT, Y_RAYAMOUNT, 1.0, 1.0);

    //CreatePlaneOld(planes, 0, vec3{ 0,0,1 }, vec3{ 0,0,-1 }, 2, 2);
    CreatePlaneOld(planes, 0, vec3{ 0,0,1 }, vec3{ 0,0,-1 }, 2, 2);

    //printf("\n\nCollision? -> %d\n\n", checkIfPlanesCollide(planes[0], planes[1]));

    printf("pa -> x:%.1f | y:%.1f | z:%.1f\n", planes[0].pa.x, planes[0].pa.y, planes[0].pa.z);
    printf("\npb -> x:%.1f | y:%.1f | z:%.1f\n", planes[0].pb.x, planes[0].pb.y, planes[0].pb.z);
    printf("\npc -> x:%.1f | y:%.1f | z:%.1f\n", planes[0].pc.x, planes[0].pc.y, planes[0].pc.z);
    printf("\npd -> x:%.1f | y:%.1f | z:%.1f\n", planes[0].pd.x, planes[0].pd.y, planes[0].pd.z);


    for (int i = 0; i < X_RAYAMOUNT * Y_RAYAMOUNT; i++)
    {
        printf("\nRayOrigin[%d] -> x: %.2f | y: %.2f | z: %.2f\n", i, rays[i].origin.x, rays[i].origin.y, rays[i].origin.z);
        printf("RayDirection[%d] -> x: %.2f | y: %.2f | z: %.2f\n", i, rays[i].direction.x, rays[i].direction.y, rays[i].direction.z);
        printf("RayEnergy[%d] -> %.2f\n", i, rays[i].energy);
    }

    cudaMalloc((void**)&d_planes, planes_size);
    cudaMemcpy(d_planes, &planes, planes_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_rays, rays_size);
    cudaMemcpy(d_rays, &rays, rays_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_total_energy, sizeof(float));
    cudaMemcpy(d_total_energy, &total_energy, sizeof(float), cudaMemcpyHostToDevice);

    int blocks_amount = ceil(X_RAYAMOUNT * Y_RAYAMOUNT, THREADSPERBLOCK);

    receiveRaysAndPlanes <<<blocks_amount, THREADSPERBLOCK >>> (d_rays, d_planes, d_total_energy);
    cudaDeviceSynchronize();

    cudaMemcpy(&total_energy, d_total_energy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_planes);
    cudaFree(d_rays);
    cudaFree(d_total_energy);

    printf("total energy absorved: %f", total_energy);

    return 0;
}