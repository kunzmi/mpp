#include "../remap_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInstantiateInvokeRemapSrcFloat2_For(32s);
ForAllChannelsWithAlphaInstantiateInvokeRemapSrc2Float_For(32s);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(32s);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(32s);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(32s);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(32s);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(32s);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(32s);

} // namespace mpp::image::cuda
