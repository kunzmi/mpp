#include "../resize_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(8s);
InstantiateP2_ForGeomType(8s);
InstantiateP3_ForGeomType(8s);
InstantiateP4_ForGeomType(8s);

} // namespace mpp::image::cuda
