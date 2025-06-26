#if MPP_ENABLE_CUDA_BACKEND

#include "../remap_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInstantiateInvokeRemapSrcFloat2_For(32fc);
ForAllChannelsNoAlphaInstantiateInvokeRemapSrc2Float_For(32fc);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(32fc);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(32fc);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(32fc);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(32fc);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(32fc);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
