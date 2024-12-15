#include "fileIOException.h"
#include "zlibCompression.h"
#include <common/defines.h>
#include <common/exception.h>
#include <cstddef>
#include <zlib/zconf.h>
#include <zlib/zlib.h>

namespace opp::fileIO
{
ZLIBDecoder::ZLIBDecoder()
{
    const int ret = inflateInit(&mStrm);
    if (ret != Z_OK)
    {
        throw EXCEPTION("Failed to initialize ZLIB / Deflate decoder. Error code: " << ret);
    }
    mInitDone = true;
}

ZLIBDecoder::~ZLIBDecoder()
{
    if (mInitDone)
    {
        (void)inflateEnd(&mStrm);
    }
}

void ZLIBDecoder::Inflate(byte *aCompressedStream, size_t aSizeIn, byte *aDecompressed, size_t aSizeDecompressed)
{
    if (aSizeIn == 0)
    {
        return;
    }
    mStrm.avail_in = uInt(aSizeIn);
    mStrm.next_in  = aCompressedStream;

    mStrm.avail_out = uInt(aSizeDecompressed);
    mStrm.next_out  = aDecompressed;

    const int ret = inflate(&mStrm, Z_NO_FLUSH);

    switch (ret)
    {
        case Z_NEED_DICT:
        case Z_DATA_ERROR:
        case Z_STREAM_ERROR: /* state not clobbered */
        case Z_MEM_ERROR:
            throw FILEIOEXCEPTION("zlib", "ZLIB-Stream inflate decoding error. Error code: " << ret);
            break;
        default:
            break;
    }

    if (ret != Z_STREAM_END || mStrm.avail_out != 0)
    {
        throw FILEIOEXCEPTION("zlib", "Decoded stream sizes do not match. Error code: " << ret);
    }
}

ZLIBEncoder::ZLIBEncoder(int aLevel)
{
    const int ret = deflateInit(&mStrm, aLevel);
    if (ret != Z_OK)
    {
        throw EXCEPTION("Failed to initialize ZLIB / Deflate encoder. Error code: " << ret);
    }
    mInitDone = true;
}

ZLIBEncoder::~ZLIBEncoder()
{
    if (mInitDone)
    {
        (void)deflateEnd(&mStrm);
    }
}

size_t ZLIBEncoder::Deflate(byte *aUncompressedStream, size_t aSizeIn, byte *aCompressed, size_t aSizeCompressed)
{
    if (aSizeIn == 0)
    {
        return 0;
    }
    mStrm.avail_in = uInt(aSizeIn);
    mStrm.next_in  = aUncompressedStream;

    mStrm.avail_out = uInt(aSizeCompressed);
    mStrm.next_out  = aCompressed;

    const int ret = deflate(&mStrm, Z_FINISH);

    if (ret != Z_STREAM_END)
    {
        throw FILEIOEXCEPTION("zlib", "ZLIB-Stream deflate encoding error. Error code: " << ret);
    }

    return mStrm.total_out;
}
} // namespace opp::fileIO