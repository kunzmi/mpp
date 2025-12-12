#include "../resize_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32u);
InstantiateP2_ForGeomType(32u);
InstantiateP3_ForGeomType(32u);
InstantiateP4_ForGeomType(32u);

} // namespace mpp::image::cuda
