#include "../remap_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInstantiateInvokeRemapSrcFloat2_For(16u);
ForAllChannelsWithAlphaInstantiateInvokeRemapSrc2Float_For(16u);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(16u);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(16u);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(16u);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(16u);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(16u);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(16u);

} // namespace mpp::image::cuda
