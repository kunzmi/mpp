#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>

TEST_CASE("Equals", "[Dummy_Test]")
{

    CHECK(true);
}

TEST_CASE("Read", "[Dummy_Test]")
{
    std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);

    CHECK(true);
}
