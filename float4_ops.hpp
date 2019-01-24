#ifndef __FLOAT4_OPS_HPP__
#define __FLOAT4_OPS_HPP__

#include <vector_functions.h>
#include <stdio.h>
#include <math.h>

/* scalar */
__host__ __device__ __forceinline__ float4&
operator+=(float4& vec, const float& v)
{
    vec.x += v;  vec.y += v;  vec.z += v; vec.w += v; return vec;
}

__host__ __device__ __forceinline__ float4&
operator*=(float4& vec, const float& v)
{
    vec.x *= v;  vec.y *= v;  vec.z *= v; vec.w *= v; return vec;
}

__host__ __device__ __forceinline__ float4
operator*(const float& v, const float4& v1)
{
    return make_float4(v * v1.x, v * v1.y, v * v1.z, v * v1.w);
}

__host__ __device__ __forceinline__ float4
operator*(const float4& v1, const float& v)
{
    return make_float4(v1.x * v, v1.y * v, v1.z * v, v1.w * v);
}

/* vector */
__host__ __device__ __forceinline__ float4
operator+(const float4& v1, const float4& v2)
{
    return make_float4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}

__host__ __device__ __forceinline__ float4&
operator+=(float4& v1, const float4& v2)
{
    v1.x += v2.x; v1.y += v2.y; v1.z += v2.z; v1.w += v2.w; return v1;
}

__host__ __device__ __forceinline__ float4
operator-(const float4& v1, const float4& v2)
{
    return make_float4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
}

__host__ __device__ __forceinline__ float4&
operator-=(float4& v1, const float4& v2)
{
    v1.x -= v2.x; v1.y -= v2.y; v1.z -= v2.z; v1.w -= v2.w; return v1;
}

__host__ __device__ __forceinline__ float
dot(const float4& v1, const float4& v2)
{
    return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w;
}

__host__ __device__ __forceinline__ float
cos(const float4& v1, const float4& v2)
{
    return dot(v1, v2) / sqrt(dot(v1, v1)*dot(v2, v2));
}

/* functions */
__host__ __device__ __forceinline__ float
coeff(const float4& v, const int &i)
{
    if (i == 0) return v.x;
    else if (i == 1) return v.y;
    else if (i == 2) return v.z;
    else return v.w;
}

__host__ __device__ __forceinline__ float
fabs_sum(const float4 &v)
{
    return fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
}

__host__ __device__ __forceinline__ float
squared_norm(const float4 &v)
{
    return dot(v, v);
}

__host__ __device__ __forceinline__ float
norm(const float4& v)
{
    return sqrt(dot(v, v));
}

__host__ __device__ __forceinline__ float4
normalized(const float4 &v)
{
    return v * (1.0f / sqrtf(dot(v, v)));
}

__host__ __device__ __forceinline__ bool
has_NaN(const float4 &v)
{
    return isnan(v.x) || isnan(v.y) || isnan(v.z) || isnan(v.w);
}

__host__ __device__ __forceinline__ float
sum(const float4& v)
{
    return v.x + v.y + v.z + v.w;
}

__host__ __device__ __forceinline__ float
mean(const float4& v)
{
    return sum(v) / 4.f;
}

/* conversion */
__host__ __device__ __forceinline__ float4
to_float4(const float3 &v, const float &w)
{
    return make_float4(v.x, v.y, v.z, w);
}

__host__ __device__ __forceinline__ float4
to_float4(const float &v)
{
    return make_float4(v, v, v, v);
}

__host__ __device__ __forceinline__ float4
expand_last_dim(const float3 &v)
{
    return make_float4(v.x, v.y, v.z, 0);
}

/* initialization*/
__host__ __device__ __forceinline__ float4
make_float4_zeros()
{
    return make_float4(0, 0, 0, 0);
}

__host__ __device__ __forceinline__ float4
make_float4_ones()
{
    return make_float4(1.f, 1.f, 1.f, 1.f);
}

__host__ __device__ __forceinline__ float4
make_float4_nans()
{
    return make_float4(NAN, NAN, NAN, NAN);
}

__host__ __device__ __forceinline__ float4
make_float4_maxs()
{
    return make_float4(3.402823466e+38f, 3.402823466e+38f, 3.402823466e+38f, 3.402823466e+38f);
}

#endif
