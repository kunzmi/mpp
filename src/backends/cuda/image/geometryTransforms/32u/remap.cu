#include "../remap_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInstantiateInvokeRemapSrcFloat2_For(32u);
ForAllChannelsWithAlphaInstantiateInvokeRemapSrc2Float_For(32u);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(32u);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(32u);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(32u);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(32u);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(32u);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(32u);

} // namespace mpp::image::cuda
