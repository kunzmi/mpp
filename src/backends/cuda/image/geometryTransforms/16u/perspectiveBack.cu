#include "../perspectiveBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16u);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(16u);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(16u);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(16u);

} // namespace mpp::image::cuda
