#include "../perspectiveBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8u);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(8u);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(8u);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(8u);

} // namespace mpp::image::cuda
