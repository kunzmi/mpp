#include "../remap_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInstantiateInvokeRemapSrcFloat2_For(16bf);
ForAllChannelsWithAlphaInstantiateInvokeRemapSrc2Float_For(16bf);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(16bf);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(16bf);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(16bf);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(16bf);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(16bf);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(16bf);

} // namespace mpp::image::cuda
