
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>

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
        return { x * scalar, y * scalar, z * scalar };
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
    struct vec3 normal;
};

//Ray constants
const int ARC_DISTANCE = 10;
const int ARC_POINTS = 5;
const int ARC_ANGLE = 45;
const int X_RAY_AMOUNT = 40;
const int Y_RAY_AMOUNT = 20;
const float ARC_HEIGHT = 2;
const float RAY_INTER_DISTANCE = .4f;
const float RAY_GROUP_ENERGY = 1;

//Solar plane constants
const int PLANE_AMOUNT = 10;
const float PLANE_WIDTH = 3;
const float PLANE_HEIGHT = 2;
const vec3 PLANE_MIN_BOUNDS = { -4, 0, -4 };
const vec3 PLANE_MAX_BOUNDS = { 4, 4, 4 };

//Simulation handle constants
const int MAXGENERATIONLOOP = 1000;

//Device constants
const int MAX_RAY_REFLECTION = 10;
const float RAY_ABSORPTION_RATIO = .2f;
const float RAY_REFLECTION_RATIO = .1f;

#pragma region DeviceFunctions

#pragma region vec3Operations

__device__ struct vec3 d_vec3Plus(struct vec3 first, struct vec3 second)
{
    return { first.x + second.x, first.y + second.y, first.z + second.z };
}

__device__ struct vec3 d_vec3Minus(struct vec3 first, struct vec3 second)
{
    return { first.x - second.x, first.y - second.y, first.z - second.z };
}

__device__ struct vec3 d_vec3Multi(struct vec3 first, struct vec3 second)
{
    return { first.x * second.x, first.y * second.y, first.z * second.z };
}

__device__ struct vec3 d_vec3Multi(struct vec3 first, float multiplier)
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

// Function to calculate the cross product between two vectors
__device__ vec3 d_cross(struct vec3 a, struct vec3 b)
{
    vec3 result;

    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;

    return result;
}

// Function to calculate the dot product between two vectors
__device__ float d_dot(struct vec3 a, struct vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Function that returns the square distance of two points
__device__ float d_squaredDistance(struct vec3 a, struct vec3 b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

// Function to normalize a vector
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

// calculate the energy of the ray on the reflected panels along the way, returns 0 if none is hitted
__device__ float calculateRayEnergy(struct plane* d_planes, struct ray incident_ray, int current_reflection, int incident_plane_index, 
                                    int planeAmount, int maxReflection, float absorptionRatio, float reflectionRatio)
{
    float total_energy = 0;

    while(current_reflection < maxReflection)
    {
        int plane_index = -1;

        vec3 edge1, edge2, crossProductResult, pointOfIntersection, edge1CrossProduct, reflection_point;
        vec3 triangleVtx1, triangleVtx2, triangleVtx3, chosenEdge1, chosenEdge2;

        float determinantA, factorF, factorU, factorV, factorT, square_distance, chosen_factorT;
        float smallest_distance = 1000000;

        for (int i = 0; i < planeAmount; i++)
        {
            if (i == incident_plane_index)
            {
                continue;
            }

            for (int j = 0; j < 2; j++)
            {
                triangleVtx1 = d_planeGet(d_planes[i], (j * 2) % 4);
                triangleVtx2 = d_planeGet(d_planes[i], (1 + j * 2) % 4);
                triangleVtx3 = d_planeGet(d_planes[i], (2 + j * 2) % 4);

                edge1 = d_vec3Minus(triangleVtx2, triangleVtx1);
                edge2 = d_vec3Minus(triangleVtx3, triangleVtx1);

                crossProductResult = d_cross(incident_ray.direction, edge2);
                determinantA = d_dot(edge1, crossProductResult);

                //Check if incident ray is parallel to plane
                if (abs(determinantA) < .000001f)
                {
                    //printf("test 1\n");
                    break;
                }

                factorF = 1.0 / determinantA;
                pointOfIntersection = d_vec3Minus(incident_ray.origin, triangleVtx1);
                factorU = factorF * d_dot(pointOfIntersection, crossProductResult);

                //Check if ray-triangle intersection is in-bounds
                if (factorU < 0 || factorU > 1)
                {
                    //printf("test 2\n");
                    continue;
                }

                edge1CrossProduct = d_cross(pointOfIntersection, edge1);
                factorV = factorF * d_dot(incident_ray.direction, edge1CrossProduct);

                //Check if ray-triangle intersection is in-bounds
                if (factorV < 0.0 || factorU + factorV > 1.0)
                {
                    //printf("test 3\n");
                    continue;
                }

                factorT = factorF * d_dot(edge2, edge1CrossProduct);

                //Check if ray-triangle intersection is not behind origin point
                if (factorT < .000001f)
                {
                    //printf("test 4\n");
                    continue;
                }

                reflection_point = d_vec3Plus(incident_ray.origin, d_vec3Multi(incident_ray.direction, factorT));
                square_distance = d_squaredDistance(incident_ray.origin, reflection_point);

                //Check if squared distance between origin and intersection point is the smallest found
                if (square_distance >= smallest_distance)
                {
                    //printf("test 5\n");
                    continue;
                }

                smallest_distance = square_distance;
                incident_plane_index = i;
                chosen_factorT = factorT;
                chosenEdge1 = edge1;
                chosenEdge2 = edge2;

                //printf("test complete\n");
                break;
            }
        }

        vec3 planeNormal = d_planes[incident_plane_index].normal;

        // First veirfy if distance is not small enough, 
        // then verify if ray is hitting plane from behind (cos is bigger than 0 (angle is bigger than 90))
        if (smallest_distance >= 1000000 || d_dot(incident_ray.direction, planeNormal) >= 0)
        {
            return total_energy;
        }

        struct ray reflected_ray;
        reflected_ray.origin = reflection_point;

        float dotProductResult = 2.0 * d_dot(incident_ray.direction, planeNormal);
        reflected_ray.direction = d_vec3Minus(incident_ray.direction, d_vec3Multi(planeNormal, dotProductResult));

        // Calculating the absorved and the reflected energy
        float reflected_energy = incident_ray.energy * reflectionRatio;
        float absorved_energy = (incident_ray.energy - reflected_energy) * absorptionRatio;
        reflected_ray.energy = reflected_energy;

        total_energy += absorved_energy;

        // Update variables for the next iteration
        incident_ray = reflected_ray;
        incident_plane_index = plane_index;
        current_reflection++;
    }

    return total_energy;
}

#pragma endregion

#pragma region GlobalFunctions

__global__ void receiveRaysAndPlanes(struct ray* d_rays, struct plane* d_planes, float* d_total_energy, int rayAmount, int planeAmount, int maxReflection, float absorptionRatio, float reflectionRatio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < rayAmount; i += stride)
    {
        float absorbed_energy = calculateRayEnergy(d_planes, d_rays[i], 0, -1, planeAmount, maxReflection, absorptionRatio, reflectionRatio);
        atomicAdd(d_total_energy, absorbed_energy);

        //printf("Ray %d -> Energy until now: %f\n", i, *d_total_energy);
    }
}

#pragma endregion

#pragma region HostFuctions

#pragma region GetFunctions

float h_vec3Get(struct vec3 vec3_toget, int index)
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

void h_vec3Set(struct vec3* vec3_toset, int index, float value)
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

struct vec3 h_planeGet(struct plane plane_toget, int index)
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
struct vec3 h_normalize(struct vec3 v) {
    float len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

    // Verificar se o comprimento é não nulo para evitar divisão por zero
    if (len != 0.0f) {
        v.x /= len;
        v.y /= len;
        v.z /= len;
    }

    return v;
}

float rand_float(float min, float max)
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float range = max - min;
    return (random * range) + min;
}

int rand_int(int min, int max)
{
    int random = rand();
    int range = max - min;
    return (random % range) + min;
}

void findPerpendicularVectors(vec3 direction, vec3* directionHeight, vec3* directionWidth)
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

void CreateSetsOfRays(struct ray* ray_array, struct vec3 direction, struct vec3 origin, int collums, int rows, float distance, float total_energy)
{
    struct vec3 directionY;
    struct vec3 directionX;

    direction = h_normalize(direction);
    findPerpendicularVectors(direction, &directionY, &directionX);

    float half_height = distance / 2;
    float half_width = distance / 2;

    vec3 current_point;
    int total_rays = 0;

    for (int i = 0; i < rows; i++)
    {
        current_point.x = origin.x + (directionY.x * half_height * (rows - 1 - (i * 2)) + directionX.x * half_width * (collums - 1) * -1);
        current_point.y = origin.y + (directionY.y * half_height * (rows - 1 - (i * 2)) + directionX.y * half_width * (collums - 1) * -1);
        current_point.z = origin.z + (directionY.z * half_height * (rows - 1 - (i * 2)) + directionX.z * half_width * (collums - 1) * -1);

        for (int j = 0; j < collums; j++)
        {
            ray_array[total_rays].origin = current_point;
            ray_array[total_rays].direction = direction;

            current_point.x += directionX.x * distance;
            current_point.y += directionX.y * distance;
            current_point.z += directionX.z * distance;

            total_rays++;
        }
    }

    float energy_per_ray = total_energy / total_rays;

    for (int i = 0; i < total_rays; i++)
    {
        ray_array[i].energy = energy_per_ray;
    }
}

void CalculateVectorByAngle(float angle, vec3* vector)
{
    // Converte o ângulo de graus para radianos
    float radianAngle = angle * (M_PI / 180.0);

    // Calcula as componentes do vetor
    vector->x = cos(radianAngle);
    vector->y = 0;
    vector->z = sin(radianAngle);
}

struct vec3 GetRaySphereIntersection(float sphereRadius, struct vec3 rayDirection)
{
    struct vec3 result;

    // Coeficientes da equação quadrática
    float a = pow(rayDirection.x, 2) + pow(rayDirection.y, 2) + pow(rayDirection.z, 2);
    float c = -pow(sphereRadius, 2);

    // Discriminante da equação quadrática
    float discriminant = -4 * a * c;

    // Calcula os pontos de interseção
    float t1 = sqrt(discriminant) / (2 * a);

    // Calcula as coordenadas do ponto de interseção
    result.x = t1 * rayDirection.x;
    result.y = t1 * rayDirection.y;
    result.z = t1 * rayDirection.z;

    return result;
}

void FindArcPoints(double radius, vec3 points[])
{
    int pTotal = ARC_POINTS + 1;
    double angulo = 0.0;
    double anguloIncremento = M_PI / pTotal;

    for (int i = 0; i < pTotal - 1; i++)
    {
        angulo += anguloIncremento;
        points[i].x = radius;
        points[i].y = radius * sin(angulo) + ARC_HEIGHT;
        points[i].z = radius * cos(angulo);
    }
}

void CreateSetsOfRayPlanes(struct ray* ray_array)
{
    vec3 rayDirection;

    CalculateVectorByAngle(ARC_ANGLE, &rayDirection);
    vec3 intersectionPoint = GetRaySphereIntersection(ARC_DISTANCE, rayDirection);

    float radius = intersectionPoint.x;
    float half_distance = RAY_INTER_DISTANCE / 2;

    vec3 arcPoints[ARC_POINTS];
    FindArcPoints(radius, arcPoints);

    vec3 directionY;
    vec3 directionX;
    vec3 pointDirection;
    vec3 current_point;

    int total_rays;
    float energy_per_ray;

    for (int i = 0; i < ARC_POINTS; i++)
    {
        pointDirection = h_normalize({ -arcPoints[i].x, -arcPoints[i].y + ARC_HEIGHT, -arcPoints[i].z });
        findPerpendicularVectors(pointDirection, &directionY, &directionX);

        total_rays = 0;

        for (int j = 0; j < Y_RAY_AMOUNT; j++)
        {
            current_point.x = arcPoints[i].x + (directionY.x * half_distance * (Y_RAY_AMOUNT - 1 - (floor(total_rays / X_RAY_AMOUNT) * 2)) + directionX.x * half_distance * (X_RAY_AMOUNT - 1) * -1);
            current_point.y = arcPoints[i].y + (directionY.y * half_distance * (Y_RAY_AMOUNT - 1 - (floor(total_rays / X_RAY_AMOUNT) * 2)) + directionX.y * half_distance * (X_RAY_AMOUNT - 1) * -1);
            current_point.z = arcPoints[i].z + (directionY.z * half_distance * (Y_RAY_AMOUNT - 1 - (floor(total_rays / X_RAY_AMOUNT) * 2)) + directionX.z * half_distance * (X_RAY_AMOUNT - 1) * -1);

            for (int k = 0; k < X_RAY_AMOUNT; k++)
            {
                ray_array[i * ARC_POINTS + total_rays].origin = current_point;
                ray_array[i * ARC_POINTS + total_rays].direction = pointDirection;

                current_point.x += directionX.x * RAY_INTER_DISTANCE;
                current_point.y += directionX.y * RAY_INTER_DISTANCE;
                current_point.z += directionX.z * RAY_INTER_DISTANCE;

                total_rays++;
            }
        }

        energy_per_ray = RAY_GROUP_ENERGY / total_rays;

        for (int j = 0; j < total_rays; j++)
        {
            ray_array[i * ARC_POINTS + j].energy = energy_per_ray;
        }
    }
}

#pragma region host vector triangle intersection functions

vec3 h_cross(struct vec3 a, struct vec3 b)
{
    vec3 result;

    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;

    return result;
}

float h_dot(struct vec3 a, struct vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Check for line-triangle intersection
bool checkIntersectionLineTriangle(struct vec3 lineStart, struct vec3 lineEnd, struct vec3 triangleVtx1, struct vec3 triangleVtx2, struct vec3 triangleVtx3)
{
    struct vec3 triangleEdge1, triangleEdge2, lineDirection, lineToStart, intersectionPoint, lineToTriangleEdge1CrossProduct;
    float determinantA, factorF, factorU, factorV, factorT;

    triangleEdge1 = triangleVtx2 - triangleVtx1;
    triangleEdge2 = triangleVtx3 - triangleVtx1;
    lineDirection = lineEnd - lineStart;

    determinantA = h_dot(triangleEdge1, h_cross(lineDirection, triangleEdge2));

    //Check if line is parallel to plane
    if (abs(determinantA) < .000001f)
    {
        return false;
    }

    factorF = 1 / determinantA;
    lineToStart = lineStart = triangleVtx1;

    factorU = factorF * h_dot(lineToStart, h_cross(lineDirection, triangleEdge2));

    //Check if line-triangle intersection is in-bounds
    if (factorU < 0.0 || factorU > 1.0)
    {
        return false;
    }

    lineToTriangleEdge1CrossProduct = h_cross(lineToStart, triangleEdge1);
    factorV = factorF * h_dot(lineDirection, lineToTriangleEdge1CrossProduct);

    //Check if line-triangle intersection is in-bounds
    if (factorV < 0.0 || factorU + factorV > 1.0)
    {
        return false;
    }

    factorT = factorF * h_dot(triangleEdge2, lineToTriangleEdge1CrossProduct);

    //Check if line-triangle intersection is not behind origin point
    if (factorT > .000001f && factorT < 1.0)
    {
        return true;
    }

    return false;
}

#pragma endregion

#pragma region planeCreation

//Divide the first plane into vectors and the second into triangles, checking for intersections
bool checkIfPlanesCollide(struct plane plane1, struct plane plane2)
{
    struct vec3 v1;
    struct vec3 v2;

    struct vec3 t1 = h_planeGet(plane2, 0);
    struct vec3 t2 = h_planeGet(plane2, 1);
    struct vec3 t3 = h_planeGet(plane2, 2);
    struct vec3 t4 = h_planeGet(plane2, 3);

    for (int i = 0; i < 4; i++)
    {
        v1 = h_planeGet(plane1, i);
        v2 = h_planeGet(plane1, (i + 1) % 4);

        if (checkIntersectionLineTriangle(v1, v2, t1, t2, t3) || checkIntersectionLineTriangle(v1, v2, t3, t4, t1))
        {
            return true;
        }
    }

    return false; // Não há interseção
}

bool checkIfPlaneIsInBounds(struct plane plane_to_check, struct vec3 positive_limit, struct vec3 negative_limit)
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

bool checkIfPlaneIsValid(struct plane plane_to_check, struct plane* plane_array, int idx_to_ignore, int plane_amount, struct vec3 positive_limit, struct vec3 negative_limit)
{
    for (int i = 0; i < plane_amount; i++)
    {
        if (idx_to_ignore == i)
        {
            continue;
        }

        if (checkIfPlanesCollide(plane_array[i], plane_to_check) || !checkIfPlaneIsInBounds(plane_to_check, positive_limit, negative_limit))
        {
            return false;
        }
    }

    return true;
}

struct plane CreatePlane(struct vec3 origin, struct vec3 direction, float plane_width, float plane_height)
{
    struct vec3 directionY;
    struct vec3 directionX;

    direction = h_normalize(direction);
    findPerpendicularVectors(direction, &directionY, &directionX);

    struct plane new_plane;

    new_plane.pa.x = origin.x + (directionY.x * plane_height / 2) + (directionX.x * plane_width / 2);
    new_plane.pa.y = origin.y + (directionY.y * plane_height / 2) + (directionX.y * plane_width / 2);
    new_plane.pa.z = origin.z + (directionY.z * plane_height / 2) + (directionX.z * plane_width / 2);

    new_plane.pb.x = origin.x + (directionY.x * plane_height / 2) - (directionX.x * plane_width / 2);
    new_plane.pb.y = origin.y + (directionY.y * plane_height / 2) - (directionX.y * plane_width / 2);
    new_plane.pb.z = origin.z + (directionY.z * plane_height / 2) - (directionX.z * plane_width / 2);

    new_plane.pc.x = origin.x - (directionY.x * plane_height / 2) - (directionX.x * plane_width / 2);
    new_plane.pc.y = origin.y - (directionY.y * plane_height / 2) - (directionX.y * plane_width / 2);
    new_plane.pc.z = origin.z - (directionY.z * plane_height / 2) - (directionX.z * plane_width / 2);

    new_plane.pd.x = origin.x - (directionY.x * plane_height / 2) + (directionX.x * plane_width / 2);
    new_plane.pd.y = origin.y - (directionY.y * plane_height / 2) + (directionX.y * plane_width / 2);
    new_plane.pd.z = origin.z - (directionY.z * plane_height / 2) + (directionX.z * plane_width / 2);

    return new_plane;
}

void CreateSetsOfPlanes(struct plane* plane_array, int plane_amount, struct vec3 positive_limit, struct vec3 negative_limit, float plane_width, float plane_height)
{
    struct vec3 direction;
    struct vec3 point;

    int rejection_limit = 100;
    float board_thickness = (plane_width + plane_height) / 2;

    for (int i = 0; i < plane_amount; i++)
    {
        direction = h_normalize(vec3{ rand_float(0, 1), rand_float(0, 1), rand_float(0, 1) });
        float soft_clamp_x = rand_float(negative_limit.x + board_thickness, positive_limit.x - board_thickness);
        float soft_clamp_y = rand_float(negative_limit.y + board_thickness, positive_limit.y - board_thickness);
        float soft_clamp_z = rand_float(negative_limit.z + board_thickness, positive_limit.z - board_thickness);

        point = { soft_clamp_x ,soft_clamp_y ,soft_clamp_z };

        struct plane new_plane = CreatePlane(point, direction, plane_width, plane_height);

        if (!checkIfPlaneIsValid(new_plane, plane_array, -1, i, positive_limit, negative_limit))
        {
            rejection_limit--;

            if (rejection_limit == 0)
            {
                break;
            }

            i--;
            continue;
        }

        new_plane.normal = direction;
        plane_array[i] = new_plane;
    }
}

#pragma endregion

void PerturbCurrentPlanesLayout(struct plane* plane_array, int plane_amount, struct vec3 positive_limit, struct vec3 negative_limit, float plane_width, float plane_height)
{
    struct vec3 direction;
    struct vec3 center_poimt;

    int rejection_limit = 100;
    int rand_idx;

    for (int i = 0; i < rejection_limit; i++)
    {
        rand_idx = rand_int(0, plane_amount);

        direction = h_normalize(vec3{ rand_float(0, 1), rand_float(0, 1), rand_float(0, 1) });
        center_poimt = { rand_float(negative_limit.x, positive_limit.x), rand_float(negative_limit.y, positive_limit.y), rand_float(negative_limit.z, positive_limit.z) };

        struct plane new_plane = CreatePlane(center_poimt, direction, plane_width, plane_height);

        if (checkIfPlaneIsValid(new_plane, plane_array, rand_idx, plane_amount, positive_limit, negative_limit))
        {
            plane_array[rand_idx] = new_plane;
            break;
        }
    }
}

// Alloc GPU memory space and transferer the data 
void copyDataToGPU(struct ray h_rays[ARC_POINTS * X_RAY_AMOUNT * Y_RAY_AMOUNT], struct plane h_planes[PLANE_AMOUNT], struct ray** d_rays, struct plane** d_planes)
{
    cudaMalloc(d_rays, ARC_POINTS * X_RAY_AMOUNT * Y_RAY_AMOUNT * sizeof(struct ray));
    cudaMalloc(d_planes, PLANE_AMOUNT * sizeof(struct plane));

    cudaMemcpy(*d_rays, h_rays, ARC_POINTS * X_RAY_AMOUNT * Y_RAY_AMOUNT * sizeof(struct ray), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_planes, h_planes, PLANE_AMOUNT * sizeof(struct plane), cudaMemcpyHostToDevice);
}

#pragma endregion

int main()
{
    clock_t start_time = clock();

    srand(time(NULL));
    int rayGroupSize = X_RAY_AMOUNT * Y_RAY_AMOUNT;

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    struct plane h_planes[PLANE_AMOUNT];
    struct plane* d_planes;
    int planes_size = sizeof(struct plane) * PLANE_AMOUNT;

    struct ray h_rays[ARC_POINTS * X_RAY_AMOUNT * Y_RAY_AMOUNT];
    struct ray* d_rays;
    int rays_size = sizeof(struct ray) * ARC_POINTS * rayGroupSize;

    float current_energy;
    float* d_simulation_energy;

    CreateSetsOfRayPlanes(h_rays);
    CreateSetsOfPlanes(h_planes, PLANE_AMOUNT, PLANE_MAX_BOUNDS, PLANE_MIN_BOUNDS, PLANE_WIDTH, PLANE_HEIGHT);

    copyDataToGPU(h_rays, h_planes, &d_rays, &d_planes);
    cudaMalloc((void**)&d_simulation_energy, sizeof(float));

    struct plane best_planes[PLANE_AMOUNT];
    float best_energy = 0;
    float energyDelta;

    float temperature = 1000;
    float coolingRate = .95;

    // Configuração do número de blocos e threads por bloco
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((rayGroupSize + threadsPerBlock.x - 1) * numberOfSMs / 256);

    for (int j = 0; j < PLANE_AMOUNT; j++)
    {
        best_planes[j] = h_planes[j];
    }

    for (int i = 0; i < MAXGENERATIONLOOP; i++)
    {
        current_energy = 0;
        cudaMemcpy(d_simulation_energy, &current_energy, sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        receiveRaysAndPlanes << <blocksPerGrid, threadsPerBlock >> > (d_rays, d_planes, d_simulation_energy, rayGroupSize * ARC_POINTS,
            PLANE_AMOUNT, MAX_RAY_REFLECTION, RAY_ABSORPTION_RATIO, RAY_REFLECTION_RATIO);

        cudaDeviceSynchronize();
        cudaMemcpy(&current_energy, d_simulation_energy, sizeof(float), cudaMemcpyDeviceToHost);

        printf("Current Generation: %d -> U Energy: %f -> A Energy: %f - ", i, current_energy, current_energy / ARC_POINTS);
        if (current_energy > best_energy)
        {
            printf("Optimized! -> old: %f , new: %f\n", best_energy, current_energy);
            best_energy = current_energy;

            for (int j = 0; j < PLANE_AMOUNT; j++)
            {
                best_planes[j] = h_planes[j];
            }
        }
        else
        {
            printf("Not optimized\n");
            for (int j = 0; j < PLANE_AMOUNT; j++)
            {
                h_planes[j] = best_planes[j];
            }
        }

        PerturbCurrentPlanesLayout(h_planes, PLANE_AMOUNT, PLANE_MAX_BOUNDS, PLANE_MIN_BOUNDS, PLANE_WIDTH, PLANE_HEIGHT);
        cudaMemcpy(d_planes, h_planes, PLANE_AMOUNT * sizeof(struct plane), cudaMemcpyHostToDevice);
    }

    cudaFree(d_rays);
    cudaFree(d_planes);
    cudaFree(d_simulation_energy);

    printf("\ntotal energy: %f\n", best_energy);

    clock_t end_time = clock();
    double cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Tempo de execucao: %f segundos\n", cpu_time_used);
    return 0;
}
