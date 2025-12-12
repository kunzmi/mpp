#include "../remap_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInstantiateInvokeRemapSrcFloat2_For(16sc);
ForAllChannelsNoAlphaInstantiateInvokeRemapSrc2Float_For(16sc);

InstantiateInvokeRemapSrcP2_Float2_ForGeomType(16sc);
InstantiateInvokeRemapSrcP2_2Float_ForGeomType(16sc);

InstantiateInvokeRemapSrcP3_Float2_ForGeomType(16sc);
InstantiateInvokeRemapSrcP3_2Float_ForGeomType(16sc);

InstantiateInvokeRemapSrcP4_Float2_ForGeomType(16sc);
InstantiateInvokeRemapSrcP4_2Float_ForGeomType(16sc);

} // namespace mpp::image::cuda
