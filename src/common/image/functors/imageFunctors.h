#pragma once
#include <common/defines.h>
#include <common/roundFunctor.h>
#include <common/vector_typetraits.h>

namespace opp::image
{
// base struct for all image processing functors
template <bool LoadBeforeOp> struct ImageFunctor
{
    // indicates if the functor operates inplace and that the kernel is supposed to load the destination image pixel
    // before the functor call. Also set to true if we load the full pixel for alpha channel.
    static constexpr bool DoLoadBeforeOp = LoadBeforeOp;
};
//
//// a functor using a constant value uses this functor as base
//// note: the constant is initialized on host and can casted once from SrcT to ComputeT
// template <typename ComputeT> struct ConstantFunctor
//{
//     ComputeT Constant;
//
//     ConstantFunctor(ComputeT aConstant) : Constant(aConstant)
//     {
//     }
// };
//
//// a functor using a device constant value uses this functor as base
//// note: the constant is only seen on device and must be casted from SrcT to ComputeT before usage
// template <typename SrcT> struct DevConstantFunctor
//{
//     const SrcT *RESTRICT Constant;
//
//     DevConstantFunctor(const SrcT *aConstant) : Constant(aConstant)
//     {
//     }
// };
//
//// a functor performing a scaling operation afer operation uses this functor as base
// template <typename ComputeT> struct ScaleFunctor
//{
//     remove_vector_t<ComputeT> ScaleFactor;
//
//     ScaleFunctor(remove_vector_t<ComputeT> aScaleFactor) : ScaleFactor(aScaleFactor)
//     {
//     }
// };

} // namespace opp::image
