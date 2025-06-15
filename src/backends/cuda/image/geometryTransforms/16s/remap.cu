#if OPP_ENABLE_CUDA_BACKEND

#include "../remap_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInstantiateInvokeRemapSrcFloat2_For(16s);
ForAllChannelsWithAlphaInstantiateInvokeRemapSrc2Float_For(16s);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(16s);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(16s);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(16s);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(16s);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(16s);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
