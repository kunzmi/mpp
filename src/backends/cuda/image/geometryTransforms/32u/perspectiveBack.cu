#include "../perspectiveBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32u);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(32u);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(32u);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(32u);

} // namespace mpp::image::cuda
