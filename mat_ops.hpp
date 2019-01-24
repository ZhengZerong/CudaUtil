#ifndef __MAT_OPS_HPP__
#define __MAT_OPS_HPP__

#include <vector_functions.h>
#include <stdio.h>
#include <math.h>

#include "float3_ops.hpp"

struct mat33 {
    __host__ __device__ mat33() {}
    __host__ __device__ mat33(const float3 &_a0, const float3 &_a1, const float3 &_a2) { cols[0] = _a0; cols[1] = _a1; cols[2] = _a2; }
    __host__ __device__ mat33(const float *_data)
    {
        /*_data MUST have at least 9 float elements, ctor does not check range*/
        cols[0] = make_float3(_data[0], _data[1], _data[2]);
        cols[1] = make_float3(_data[3], _data[4], _data[5]);
        cols[2] = make_float3(_data[6], _data[7], _data[8]);
    }

    __host__ __device__ float value_at(int ri, int ci) { return coeff(cols[ci], ri); }

    __host__ __device__ const float& m00() const { return cols[0].x; }
    __host__ __device__ const float& m10() const { return cols[0].y; }
    __host__ __device__ const float& m20() const { return cols[0].z; }
    __host__ __device__ const float& m01() const { return cols[1].x; }
    __host__ __device__ const float& m11() const { return cols[1].y; }
    __host__ __device__ const float& m21() const { return cols[1].z; }
    __host__ __device__ const float& m02() const { return cols[2].x; }
    __host__ __device__ const float& m12() const { return cols[2].y; }
    __host__ __device__ const float& m22() const { return cols[2].z; }

    __host__ __device__ float& m00() { return cols[0].x; }
    __host__ __device__ float& m10() { return cols[0].y; }
    __host__ __device__ float& m20() { return cols[0].z; }
    __host__ __device__ float& m01() { return cols[1].x; }
    __host__ __device__ float& m11() { return cols[1].y; }
    __host__ __device__ float& m21() { return cols[1].z; }
    __host__ __device__ float& m02() { return cols[2].x; }
    __host__ __device__ float& m12() { return cols[2].y; }
    __host__ __device__ float& m22() { return cols[2].z; }

    __host__ __device__ mat33 transpose() const
    {
        float3 row0 = make_float3(cols[0].x, cols[1].x, cols[2].x);
        float3 row1 = make_float3(cols[0].y, cols[1].y, cols[2].y);
        float3 row2 = make_float3(cols[0].z, cols[1].z, cols[2].z);
        return mat33(row0, row1, row2);
    }

    __host__ __device__ mat33 operator* (const mat33 &_mat) const
    {
        mat33 mat;
        mat.m00() = m00()*_mat.m00() + m01()*_mat.m10() + m02()*_mat.m20();
        mat.m01() = m00()*_mat.m01() + m01()*_mat.m11() + m02()*_mat.m21();
        mat.m02() = m00()*_mat.m02() + m01()*_mat.m12() + m02()*_mat.m22();
        mat.m10() = m10()*_mat.m00() + m11()*_mat.m10() + m12()*_mat.m20();
        mat.m11() = m10()*_mat.m01() + m11()*_mat.m11() + m12()*_mat.m21();
        mat.m12() = m10()*_mat.m02() + m11()*_mat.m12() + m12()*_mat.m22();
        mat.m20() = m20()*_mat.m00() + m21()*_mat.m10() + m22()*_mat.m20();
        mat.m21() = m20()*_mat.m01() + m21()*_mat.m11() + m22()*_mat.m21();
        mat.m22() = m20()*_mat.m02() + m21()*_mat.m12() + m22()*_mat.m22();
        return mat;
    }

    __host__ __device__ mat33 operator+ (const mat33 &_mat) const
    {
        mat33 mat_sum;
        mat_sum.m00() = m00() + _mat.m00();
        mat_sum.m01() = m01() + _mat.m01();
        mat_sum.m02() = m02() + _mat.m02();

        mat_sum.m10() = m10() + _mat.m10();
        mat_sum.m11() = m11() + _mat.m11();
        mat_sum.m12() = m12() + _mat.m12();

        mat_sum.m20() = m20() + _mat.m20();
        mat_sum.m21() = m21() + _mat.m21();
        mat_sum.m22() = m22() + _mat.m22();

        return mat_sum;
    }

    __host__ __device__ mat33 operator- (const mat33 &_mat) const
    {
        mat33 mat_diff;
        mat_diff.m00() = m00() - _mat.m00();
        mat_diff.m01() = m01() - _mat.m01();
        mat_diff.m02() = m02() - _mat.m02();

        mat_diff.m10() = m10() - _mat.m10();
        mat_diff.m11() = m11() - _mat.m11();
        mat_diff.m12() = m12() - _mat.m12();

        mat_diff.m20() = m20() - _mat.m20();
        mat_diff.m21() = m21() - _mat.m21();
        mat_diff.m22() = m22() - _mat.m22();

        return mat_diff;
    }

    __host__ __device__ mat33 operator-() const
    {
        mat33 mat_neg;
        mat_neg.m00() = -m00();
        mat_neg.m01() = -m01();
        mat_neg.m02() = -m02();

        mat_neg.m10() = -m10();
        mat_neg.m11() = -m11();
        mat_neg.m12() = -m12();

        mat_neg.m20() = -m20();
        mat_neg.m21() = -m21();
        mat_neg.m22() = -m22();

        return mat_neg;
    }

    __host__ __device__ mat33& operator*= (const mat33 &_mat)
    {
        *this = *this * _mat;
        return *this;
    }

    __host__ __device__ float3 operator* (const float3 &_vec) const
    {
        float x = m00()*_vec.x + m01()*_vec.y + m02()*_vec.z;
        float y = m10()*_vec.x + m11()*_vec.y + m12()*_vec.z;
        float z = m20()*_vec.x + m21()*_vec.y + m22()*_vec.z;
        return make_float3(x, y, z);
    }

    __host__ __device__ mat33 operator* (const float &_f) const
    {
        return mat33(cols[0] * _f, cols[1] * _f, cols[2] * _f);
    }

    __host__ __device__ mat33 inverse() const
    {
        /*
        Reference:
        d = (m00_*m11_*m22_ - m00_*m12_*m21_ - m01_*m10_*m22_ + m01_*m12_*m20_ + m02_*m10_*m21_ - m02_*m11_*m20_)
        ans =

        [  (m11_*m22_ - m12_*m21_)/d, -(m01_*m22_ - m02_*m21_)/d,  (m01_*m12_ - m02_*m11_)/d]
        [ -(m10_*m22_ - m12_*m20_)/d,  (m00_*m22_ - m02_*m20_)/d, -(m00_*m12_ - m02_*m10_)/d]
        [  (m10_*m21_ - m11_*m20_)/d, -(m00_*m21_ - m01_*m20_)/d,  (m00_*m11_ - m01_*m10_)/d]

        */

        float d = m00()*m11()*m22() - m00()*m12()*m21() - m01()*m10()*m22() + m01()*m12()*m20() + m02()*m10()*m21() - m02()*m11()*m20();
        mat33 r;
        r.m00() = (m11()*m22() - m12()*m21());
        r.m01() = -(m01()*m22() - m02()*m21());
        r.m02() = (m01()*m12() - m02()*m11());
        r.m10() = -(m10()*m22() - m12()*m20());
        r.m11() = (m00()*m22() - m02()*m20());
        r.m12() = -(m00()*m12() - m02()*m10());
        r.m20() = (m10()*m21() - m11()*m20());
        r.m21() = -(m00()*m21() - m01()*m20());
        r.m22() = (m00()*m11() - m01()*m10());

        return r * (1.f / d);
    }

    __host__ __device__ void set_identity()
    {
        cols[0] = make_float3(1, 0, 0);
        cols[1] = make_float3(0, 1, 0);
        cols[2] = make_float3(0, 0, 1);
    }

    __host__ __device__ static mat33 identity()
    {
        mat33 m;
        m.set_identity();
        return m;
    }

    __host__ __device__ void set_zeros()
    {
        cols[0] = make_float3(0, 0, 0);
        cols[1] = make_float3(0, 0, 0);
        cols[2] = make_float3(0, 0, 0);
    }

    __host__ __device__ static mat33 zeros()
    {
        mat33 m;
        m.set_zeros();
        return m;
    }

    __host__ __device__ void set_ones()
    {
        cols[0] = make_float3(1.f, 1.f, 1.f);
        cols[1] = make_float3(1.f, 1.f, 1.f);
        cols[2] = make_float3(1.f, 1.f, 1.f);
    }

    __host__ __device__ static mat33 ones()
    {
        mat33 m;
        m.set_ones();
        return m;
    }

    __host__ __device__ void print() const
    {
        printf("%6.6f %6.6f %6.6f \n", m00(), m01(), m02());
        printf("%6.6f %6.6f %6.6f \n", m10(), m11(), m12());
        printf("%6.6f %6.6f %6.6f \n", m20(), m21(), m22());
    }

    float3 cols[3]; /*colume major*/
}; 

/*rotation and translation*/
struct mat34 {
    __host__ __device__ mat34() {}
    __host__ __device__ mat34(const mat33 &_rot, const float3 &_trans) : rot(_rot), trans(_trans) {}

    __host__ __device__ static mat34 identity()
    {
        return mat34(mat33::identity(), make_float3(0, 0, 0));
    }

    __host__ __device__ static mat34 zeros()
    {
        return mat34(mat33::zeros(), make_float3(0, 0, 0));
    }

    __host__ __device__ mat34 operator* (const mat34 &_right_se3) const
    {
        mat34 se3;
        se3.rot = rot*_right_se3.rot;
        se3.trans = rot*_right_se3.trans + trans;
        return se3;
    }

    __host__ __device__ mat34 operator* (const float &_w) const
    {
        mat34 se3;
        se3.rot = rot;
        se3.trans = trans;

        //se3.rot.m00() *= _w*se3.rot.m00();	se3.rot.m01() *= _w*se3.rot.m01();	se3.rot.m02() *= _w*se3.rot.m02();
        //se3.rot.m10() *= _w*se3.rot.m10();	se3.rot.m11() *= _w*se3.rot.m11();	se3.rot.m12() *= _w*se3.rot.m12();
        //se3.rot.m20() *= _w*se3.rot.m20();	se3.rot.m21() *= _w*se3.rot.m21();	se3.rot.m22() *= _w*se3.rot.m22();

        //se3.trans.x *= _w*se3.trans.x;	se3.trans.y *= _w*se3.trans.y;	se3.trans.z *= _w*se3.trans.z;


        se3.rot.m00() *= _w;	se3.rot.m01() *= _w;	se3.rot.m02() *= _w;
        se3.rot.m10() *= _w;	se3.rot.m11() *= _w;	se3.rot.m12() *= _w;
        se3.rot.m20() *= _w;	se3.rot.m21() *= _w;	se3.rot.m22() *= _w;

        se3.trans.x *= _w;	se3.trans.y *= _w;	se3.trans.z *= _w;
        return se3;
    }

    __host__ __device__ mat34 operator+ (const mat34 &_T) const
    {
        mat34 se3;
        se3.rot = rot;
        se3.trans = trans;

        se3.rot.m00() += _T.rot.m00();	se3.rot.m01() += _T.rot.m01();	se3.rot.m02() += _T.rot.m02();
        se3.rot.m10() += _T.rot.m10();	se3.rot.m11() += _T.rot.m11();	se3.rot.m12() += _T.rot.m12();
        se3.rot.m20() += _T.rot.m20();	se3.rot.m21() += _T.rot.m21();	se3.rot.m22() += _T.rot.m22();

        se3.trans.x += _T.trans.x;	se3.trans.y += _T.trans.y;	se3.trans.z += _T.trans.z;

        return se3;
    }


    __host__ __device__ mat34& operator*= (const mat34 &_right_se3)
    {
        *this = *this * _right_se3;
        return *this;
    }

    __host__ __device__ float3 operator* (const float3 &_vec) const
    {
        return rot * _vec + trans;
    }

    __host__ __device__ float4 operator* (const float4 &_vec) const
    {
        float3 v3 = make_float3(_vec.x, _vec.y, _vec.z);
        float3 v3_ = rot * v3 + trans;
        return make_float4(v3_.x, v3_.y, v3_.z, _vec.w);
    }

    __host__ __device__ mat34 inverse() const
    {
        mat34 r;
        mat33 rot_inv = rot.inverse();
        r.rot = rot_inv;
        r.trans = -rot_inv * trans;
        return r;
    }

    __host__ __device__ void print() const
    {
        printf("%f %f %f %f \n", rot.m00(), rot.m01(), rot.m02(), trans.x);
        printf("%f %f %f %f \n", rot.m10(), rot.m11(), rot.m12(), trans.y);
        printf("%f %f %f %f \n", rot.m20(), rot.m21(), rot.m22(), trans.z);
    }

    mat33 rot;
    float3 trans;
};
#endif