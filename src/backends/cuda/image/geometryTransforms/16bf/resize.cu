#include "../resize_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16bf);
InstantiateP2_ForGeomType(16bf);
InstantiateP3_ForGeomType(16bf);
InstantiateP4_ForGeomType(16bf);

} // namespace mpp::image::cuda
