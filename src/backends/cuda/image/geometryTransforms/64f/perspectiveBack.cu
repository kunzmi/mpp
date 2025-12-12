#include "../perspectiveBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(64f);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(64f);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(64f);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(64f);

} // namespace mpp::image::cuda
