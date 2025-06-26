#pragma once
#include <common/defines.h>
#include <concepts>

#include "vector1.h"
#include "vector2.h"
#include "vector3.h"
#include "vector4.h"
#include "vector4A.h"

// define some common shortcuts:
namespace mpp
{
using Vec2i = Vector2<int>;
using Vec3i = Vector3<int>;
using Vec4i = Vector4<int>;

using Vec2f = Vector2<float>;
using Vec3f = Vector3<float>;
using Vec4f = Vector4<float>;

using Vec2d = Vector2<double>;
using Vec3d = Vector3<double>;
using Vec4d = Vector4<double>;
} // namespace mpp
