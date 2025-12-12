#include "../perspectiveBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8s);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(8s);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(8s);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(8s);

} // namespace mpp::image::cuda
