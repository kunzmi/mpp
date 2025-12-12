#include "../remap_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInstantiateInvokeRemapSrcFloat2_For(16f);
ForAllChannelsWithAlphaInstantiateInvokeRemapSrc2Float_For(16f);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(16f);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(16f);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(16f);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(16f);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(16f);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(16f);

} // namespace mpp::image::cuda
