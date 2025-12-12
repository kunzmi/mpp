#include "../resize_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16s);
InstantiateP2_ForGeomType(16s);
InstantiateP3_ForGeomType(16s);
InstantiateP4_ForGeomType(16s);

} // namespace mpp::image::cuda
