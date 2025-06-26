#include "file.h"
#include "fileIOException.h"
#include <algorithm>
#include <common/defines.h>
#include <common/exception.h>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <locale>
#include <string>
#include <utility>

namespace mpp::fileIO
{
FileType GuessFileTypeFromEnding(const std::filesystem::path &aFileName)
{
    std::string extension = aFileName.extension().string();
    std::locale loc;
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [&](auto &ac) { return std::tolower(ac, loc); });

    if (extension == ".tif")
    {
        return FileType::TIFF;
    }
    if (extension == ".tiff")
    {
        return FileType::TIFF;
    }

    return FileType::UNKNOWN;
}

void File::SetFileName(const std::filesystem::path &aFileName)
{
    mFileName = aFileName;
}

File::File(std::filesystem::path aFileName, bool aIsLittleEndian)
    : mFileName(std::move(aFileName)), mIsLittleEndian(aIsLittleEndian)
{
}

File &File::operator=(File &&aOther) noexcept
{
    if (!(aOther.mIsMoved))
    {
        mFileName       = std::move(aOther.mFileName);
        mIsLittleEndian = aOther.mIsLittleEndian;
        aOther.mIsMoved = true;
    }
    return *this;
}

void File::EndianSwap(char *aData, size_t aElementCount, size_t aElementSize) // NOLINT
{
    ushort *d_us  = nullptr;
    uint *d_i     = nullptr;
    ulong64 *d_ul = nullptr;

    switch (aElementSize)
    {
        case 1:
            return;
        case 2:
            // signed and unsigned short are the same
            d_us = reinterpret_cast<ushort *>(aData);
            EndianSwap(d_us, d_us + aElementCount);
            break;
        case 4:
            d_i = reinterpret_cast<uint *>(aData);
            EndianSwap(d_i, d_i + aElementCount);
            break;
        case 8:
            d_ul = reinterpret_cast<ulong64 *>(aData);
            EndianSwap(d_ul, d_ul + aElementCount);
            break;
        default:
            throw INVALIDARGUMENT(
                aElementSize,
                "Invalid data element size for Endian swap. Supported sizes are: 1, 2, 4 and 8 bytes. Got: "
                    << aElementSize);
    }
}

bool File::Exists(const std::filesystem::path &aFileName)
{
    return std::filesystem::exists(aFileName);
}
void File::ClearContent(const std::filesystem::path &aFileName)
{
    std::ofstream clearFile(aFileName, std::ofstream::trunc);
    if (!clearFile.good())
    {
        throw FILEIOEXCEPTION(aFileName, "Failed to access file for writing.");
    }
    clearFile.close();
}

} // namespace mpp::fileIO