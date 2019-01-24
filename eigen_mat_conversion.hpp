/*
* Software License Agreement (BSD License)

Copyright (c) 2019, Zerong Zheng (zzr18@mails.tsinghua.edu.cn)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef SMPL_API_EIGEN_MAT_CONVERSION_HPP_
#define SMPL_API_EIGEN_MAT_CONVERSION_HPP_

#include "Eigen/Eigen"
#include "vector_ops.hpp"

Eigen::Vector3f ToEigenV3f(const float3 &_v) {
    return Eigen::Vector3f(_v.x, _v.y, _v.z);
}
Eigen::Vector3f ToEigenV3f(const float4 &_v) {
    return Eigen::Vector3f(_v.x, _v.y, _v.z);
}
Eigen::Vector4f ToEigenV4f(const float3 &_v) {
    return Eigen::Vector4f(_v.x, _v.y, _v.z, 0.f);
}
Eigen::Vector4f ToEigenV4f(const float4 &_v) {
    return Eigen::Vector4f(_v.x, _v.y, _v.z, _v.w);
}
Eigen::Matrix3f ToEigenM3f(const mat33 &_m) {
    Eigen::Matrix3f res;
    res(0, 0) = _m.m00(); res(0, 1) = _m.m01(); res(0, 2) = _m.m02();
    res(1, 0) = _m.m10(); res(1, 1) = _m.m11(); res(1, 2) = _m.m12();
    res(2, 0) = _m.m20(); res(2, 1) = _m.m21(); res(2, 2) = _m.m22();
    return res;
}
Eigen::Matrix4f ToEigenM4f(const mat34 &_m) {
    Eigen::Matrix3f rot = ToEigenM3f(_m.rot);
    Eigen::Vector3f trans = ToEigenV3f(_m.trans);
    Eigen::Matrix4f m;
    m.topLeftCorner(3, 3) = rot;
    m.topRightCorner(3, 1) = trans;
    return m;
}

float3 ToFloat3(const Eigen::Vector3f &_v) {
    return make_float3(_v[0], _v[1], _v[2]);
}
float3 ToFloat3(const Eigen::Vector4f &_v) {
    return make_float3(_v[0], _v[1], _v[2]);
}
float4 ToFloat4(const Eigen::Vector3f &_v) {
    return make_float4(_v[0], _v[1], _v[2], 0.f);
}
float4 ToFloat4(const Eigen::Vector4f &_v) {
    return make_float4(_v[0], _v[1], _v[2], _v[3]);
}
mat33 ToMat33(const Eigen::Matrix3f &_m) {
    mat33 res = mat33::zeros();
    res.m00() = _m(0, 0); res.m01() = _m(0, 1); res.m02() = _m(0, 2);
    res.m10() = _m(1, 0); res.m11() = _m(1, 1); res.m12() = _m(1, 2);
    res.m20() = _m(2, 0); res.m21() = _m(2, 1); res.m22() = _m(2, 2);
    return res;
}
mat34 ToMat34(const Eigen::Matrix4f &_m) {
    mat34 res;
    res.rot = ToMat33(_m.topLeftCorner(3, 3));
    Eigen::Vector3f trans = _m.col(3).topRows(3);
    res.trans = ToFloat3(trans);
    return res;
}

#endif