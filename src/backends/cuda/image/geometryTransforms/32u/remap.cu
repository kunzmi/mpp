#if OPP_ENABLE_CUDA_BACKEND

#include "../remap_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInstantiateInvokeRemapSrcFloat2_For(32u);
ForAllChannelsWithAlphaInstantiateInvokeRemapSrc2Float_For(32u);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(32u);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(32u);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(32u);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(32u);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(32u);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(32u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
