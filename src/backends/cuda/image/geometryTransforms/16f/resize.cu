#include "../resize_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16f);
InstantiateP2_ForGeomType(16f);
InstantiateP3_ForGeomType(16f);
InstantiateP4_ForGeomType(16f);

} // namespace mpp::image::cuda
