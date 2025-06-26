#if MPP_ENABLE_CUDA_BACKEND

#include "../remap_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInstantiateInvokeRemapSrcFloat2_For(8s);
ForAllChannelsWithAlphaInstantiateInvokeRemapSrc2Float_For(8s);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(8s);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(8s);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(8s);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(8s);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(8s);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
