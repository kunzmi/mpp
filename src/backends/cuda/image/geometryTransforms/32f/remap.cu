#include "../remap_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInstantiateInvokeRemapSrcFloat2_For(32f);
ForAllChannelsWithAlphaInstantiateInvokeRemapSrc2Float_For(32f);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(32f);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(32f);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(32f);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(32f);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(32f);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(32f);

} // namespace mpp::image::cuda
