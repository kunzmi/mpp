#include "../resize_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(64f);
InstantiateP2_ForGeomType(64f);
InstantiateP3_ForGeomType(64f);
InstantiateP4_ForGeomType(64f);

} // namespace mpp::image::cuda
