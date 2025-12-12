#include "../perspectiveBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16s);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(16s);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(16s);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(16s);

} // namespace mpp::image::cuda
