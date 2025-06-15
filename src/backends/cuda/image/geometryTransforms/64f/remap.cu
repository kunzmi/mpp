#if OPP_ENABLE_CUDA_BACKEND

#include "../remap_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInstantiateInvokeRemapSrcFloat2_For(64f);
ForAllChannelsWithAlphaInstantiateInvokeRemapSrc2Float_For(64f);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(64f);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(64f);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(64f);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(64f);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(64f);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
