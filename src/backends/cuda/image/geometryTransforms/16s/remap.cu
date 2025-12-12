#include "../remap_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInstantiateInvokeRemapSrcFloat2_For(16s);
ForAllChannelsWithAlphaInstantiateInvokeRemapSrc2Float_For(16s);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(16s);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(16s);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(16s);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(16s);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(16s);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(16s);

} // namespace mpp::image::cuda
