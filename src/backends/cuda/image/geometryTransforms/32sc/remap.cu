#if OPP_ENABLE_CUDA_BACKEND

#include "../remap_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInstantiateInvokeRemapSrcFloat2_For(32sc);
ForAllChannelsNoAlphaInstantiateInvokeRemapSrc2Float_For(32sc);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(32sc);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(32sc);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(32sc);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(32sc);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(32sc);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
