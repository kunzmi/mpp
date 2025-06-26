#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/sizePitched.h>
#include <common/vector2.h>
#include <cstddef>
#include <sstream>
#include <vector>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;

TEST_CASE("SizePitched", "[Common.Image]")
{
    Size2D t1(100, 200);
    size_t pitch = 1024;

    SizePitched sp(t1, pitch);

    CHECK(sp.Size() == t1);
    CHECK(sp.Pitch() == pitch);
}