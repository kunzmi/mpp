#include "../resize_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32f);
InstantiateP2_ForGeomType(32f);
InstantiateP3_ForGeomType(32f);
InstantiateP4_ForGeomType(32f);

} // namespace mpp::image::cuda
