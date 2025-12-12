#include "../perspectiveBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(32fc);
InstantiateInvokePerspectiveBackSrcP2_ForGeomType(32fc);
InstantiateInvokePerspectiveBackSrcP3_ForGeomType(32fc);
InstantiateInvokePerspectiveBackSrcP4_ForGeomType(32fc);

} // namespace mpp::image::cuda
