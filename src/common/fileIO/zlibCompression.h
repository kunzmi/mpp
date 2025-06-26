#pragma once

#include <common/defines.h>
#include <zlib/zlib.h>

namespace mpp::fileIO
{

/// <summary>
/// ZLIB / Deflate decoder utlity class
/// </summary>
class ZLIBDecoder
{
  public:
    ZLIBDecoder();

    ~ZLIBDecoder();

    ZLIBDecoder(const ZLIBDecoder &) = delete;
    ZLIBDecoder(ZLIBDecoder &&)      = delete;

    ZLIBDecoder &operator=(const ZLIBDecoder &) = delete;
    ZLIBDecoder &operator=(ZLIBDecoder &&)      = delete;

    void Inflate(byte *aCompressedStream, size_t aSizeIn, byte *aDecompressed, size_t aSizeDecompressed);

  private:
    z_stream mStrm{nullptr};
    bool mInitDone{false};
};

/// <summary>
/// ZLIB / Inflate encoder utlity class
/// </summary>
class ZLIBEncoder
{
  public:
    explicit ZLIBEncoder(int aLevel);

    ~ZLIBEncoder();

    ZLIBEncoder(const ZLIBEncoder &) = delete;
    ZLIBEncoder(ZLIBEncoder &&)      = delete;

    ZLIBEncoder &operator=(const ZLIBEncoder &) = delete;
    ZLIBEncoder &operator=(ZLIBEncoder &&)      = delete;

    /// <summary>
    /// Compresses an byte* input array to byte* output array. Returns the final size of aDecompressed.
    /// </summary>
    size_t Deflate(byte *aUncompressedStream, size_t aSizeIn, byte *aCompressed, size_t aSizeCompressed);

  private:
    z_stream mStrm{nullptr};
    bool mInitDone{false};
};

} // namespace mpp::fileIO