#include "../resize_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32s);
InstantiateP2_ForGeomType(32s);
InstantiateP3_ForGeomType(32s);
InstantiateP4_ForGeomType(32s);

} // namespace mpp::image::cuda
