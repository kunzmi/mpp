#pragma once
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <concepts>

#ifndef __restrict__
#define __restrict__
#define myDef
#endif

#include <common/disableWarningsBegin.h>

namespace opp::image
{

//
//
// Inplace functors: One SrcDst array and zero or more additional src arrays

// With float scaling of result

} // namespace opp::image

#include <common/disableWarningsEnd.h>

#ifdef myDef
#undef __restrict__
#undef myDef
#endif // myDef