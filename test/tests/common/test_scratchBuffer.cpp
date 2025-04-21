#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/scratchBuffer.h>
#include <common/scratchBufferException.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

using namespace opp;
using namespace Catch;

template <typename T> void testCheck(T &aBuffer, size_t aSize)
{
    CHECK_BUFFER_SIZE(aBuffer, aSize);
}

TEST_CASE("ScratchBuffer", "[Common]")
{
    int *ptr0                   = nullptr;
    std::array<size_t, 1> sizes = {128};
    ScratchBuffer<int> buffer1(ptr0, sizes);
    ScratchBuffer<int> buffer2(ptr0 + 16, sizes);

    CHECK(buffer1.GetTotalBufferSize() == 128 * sizeof(int) + 256);
    CHECK(buffer2.GetTotalBufferSize() == 128 * sizeof(int) + 256);
    CHECK(reinterpret_cast<size_t>(buffer1.Get<0>()) == 0ull);
    CHECK(reinterpret_cast<size_t>(buffer2.Get<0>()) == 256ull);

    CHECK_NOTHROW(testCheck(buffer1, 768));
    CHECK_THROWS_AS(testCheck(buffer1, 512), opp::ScratchBufferException);
}
