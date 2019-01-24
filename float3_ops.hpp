#ifndef __FLOAT3_OPS_HPP__
#define __FLOAT3_OPS_HPP__

#include <vector_functions.h>
#include <stdio.h>
#include <math.h>

/* scalar */
__host__ __device__ __forceinline__ float3&
operator+=(float3& vec, const float& v)
{
    vec.x += v;  vec.y += v;  vec.z += v; return vec;
}

__host__ __device__ __forceinline__ float3&
operator*=(float3& vec, const float& v)
{
    vec.x *= v;  vec.y *= v;  vec.z *= v; return vec;
}

__host__ __device__ __forceinline__ float3
operator*(const float& v, const float3& v1)
{
    return make_float3(v * v1.x, v * v1.y, v * v1.z);
}

__host__ __device__ __forceinline__ float3
operator*(const float3& v1, const float& v)
{
    return make_float3(v1.x * v, v1.y * v, v1.z * v);
}

/* vector */
__host__ __device__ __forceinline__ float3
operator+(const float3& v1, const float3& v2)
{
    return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ __forceinline__ float3&
operator+=(float3& v1, const float3& v2)
{
    v1.x += v2.x; v1.y += v2.y; v1.z += v2.z; return v1;
}

__host__ __device__ __forceinline__ float3
operator-(const float3& v1, const float3& v2)
{
    return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__host__ __device__ __forceinline__ float3&
operator-=(float3& v1, const float3& v2)
{
    v1.x -= v2.x; v1.y -= v2.y; v1.z -= v2.z; return v1;
}

__host__ __device__ __forceinline__ float
dot(const float3& v1, const float3& v2)
{
    return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
}

__host__ __device__ __forceinline__ float3
cross(const float3& v1, const float3& v2)
{
    return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

__host__ __device__ __forceinline__ float
cos(const float3& v1, const float3& v2)
{
    return dot(v1, v2) / sqrt(dot(v1, v1)*dot(v2, v2));
}

/* functions */
__host__ __device__ __forceinline__ float
coeff(const float3& v, const int &i)
{
    if (i == 0) return v.x;
    else if (i == 1) return v.y;
    else return v.z;
}

__host__ __device__ __forceinline__ float
fabs_sum(const float3 &v)
{
    return fabsf(v.x) + fabsf(v.y) + fabsf(v.z);
}

__host__ __device__ __forceinline__ float
squared_norm(const float3 &v)
{
    return dot(v, v);
}

__host__ __device__ __forceinline__ float
norm(const float3& v)
{
    return sqrt(dot(v, v));
}

__host__ __device__ __forceinline__ float3
normalized(const float3 &v)
{
    return v * (1.0f / sqrtf(dot(v, v)));
}

__host__ __device__ __forceinline__ bool
has_NaN(const float3 &v)
{
    return isnan(v.x) || isnan(v.y) || isnan(v.z);
}

__host__ __device__ __forceinline__ float
sum(const float3& v)
{
    return v.x + v.y + v.z;
}

__host__ __device__ __forceinline__ float
mean(const float3& v)
{
    return sum(v) / 3.f;
}

/* conversion */
__host__ __device__ __forceinline__ float3
to_float3(const float4 &v)
{
    return make_float3(v.x, v.y, v.z);
}

__host__ __device__ __forceinline__ float3
to_float3(const float &v)
{
    return make_float3(v, v, v);
}

__host__ __device__ __forceinline__ float3
discard_last_dim(const float4 &v)
{
    return make_float3(v.x, v.y, v.z);
}

/* initialization*/
__host__ __device__ __forceinline__ float3
make_float3_zeros()
{
    return make_float3(0, 0, 0);
}

__host__ __device__ __forceinline__ float3
make_float3_ones()
{
    return make_float3(1.f, 1.f, 1.f);
}

__host__ __device__ __forceinline__ float3
make_float3_nans()
{
    return make_float3(NAN, NAN, NAN);
}

__host__ __device__ __forceinline__ float3
make_float3_maxs()
{
    return make_float3(3.402823466e+38f, 3.402823466e+38f, 3.402823466e+38f);
}

#endif
