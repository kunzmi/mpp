#pragma once

// this header file includes all _impl.h headers for vector types needed for compilation on non-host devices, i.e. that
// don't compile the cpp files.

#include "bfloat16_impl.h"
#include "half_fp16_impl.h"

#include "complex_impl.h"

#include "vector1_impl.h"
#include "vector2_impl.h"
#include "vector3_impl.h"
#include "vector4_impl.h"
#include "vector4A_impl.h"
