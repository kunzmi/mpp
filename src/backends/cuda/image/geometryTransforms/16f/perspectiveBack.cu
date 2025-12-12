#include "../perspectiveBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16f);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(16f);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(16f);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(16f);

} // namespace mpp::image::cuda
