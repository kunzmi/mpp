#include "../remap_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInstantiateInvokeRemapSrcFloat2_For(8u);
ForAllChannelsWithAlphaInstantiateInvokeRemapSrc2Float_For(8u);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(8u);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(8u);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(8u);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(8u);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(8u);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(8u);

} // namespace mpp::image::cuda
