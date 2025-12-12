#include "../resize_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(16sc);
InstantiateP2_ForGeomType(16sc);
InstantiateP3_ForGeomType(16sc);
InstantiateP4_ForGeomType(16sc);

} // namespace mpp::image::cuda
