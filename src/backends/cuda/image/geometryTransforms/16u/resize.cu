#include "../resize_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16u);
InstantiateP2_ForGeomType(16u);
InstantiateP3_ForGeomType(16u);
InstantiateP4_ForGeomType(16u);

} // namespace mpp::image::cuda
