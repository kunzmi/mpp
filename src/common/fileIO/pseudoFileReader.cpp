#include "pseudoFileReader.h"
#include <common/fileIO/file.h>
#include <common/fileIO/fileReader.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/vector3.h>
#include <cstddef>
#include <filesystem>
#include <istream>
#include <memory>

namespace mpp::fileIO
{
PseudoFileReader::PseudoFileReader() : File("", true)
{
}

PseudoFileReader::PseudoFileReader(std::shared_ptr<std::istream> &aStream)
    : File("", true), FileReader(aStream) // File is actually not initialized here
{
}

void PseudoFileReader::OpenAndRead()
{
}

void PseudoFileReader::OpenAndReadHeader()
{
}

bool PseudoFileReader::TryToOpenAndReadHeader() noexcept
{
    return true;
}

mpp::image::PixelTypeEnum PseudoFileReader::GetDataType() const
{
    return mpp::image::PixelTypeEnum::Unknown;
}

Vector3<int> PseudoFileReader::Size() const
{
    return {};
}

mpp::image::Size2D PseudoFileReader::SizePlane() const
{
    return {};
}

double PseudoFileReader::PixelSize() const
{
    return 0;
}

void *PseudoFileReader::Data()
{
    return nullptr;
}

void *PseudoFileReader::Data(size_t /*aIdx*/)
{
    return nullptr;
}

size_t PseudoFileReader::DataSize() const
{
    return 0;
}

size_t PseudoFileReader::GetImageSizeInBytes() const
{
    return 0;
}

void PseudoFileReader::ReadSlice(size_t /*aIdx*/)
{
}
void PseudoFileReader::ReadSlices(size_t /*aStartIdx*/, size_t /*aSliceCount*/)
{
}
void PseudoFileReader::ReadSlice(void * /*aData*/, size_t /*aIdx*/)
{
}
void PseudoFileReader::ReadSlices(void * /*aData*/, size_t /*aStartIdx*/, size_t /*aSliceCount*/)
{
}
void PseudoFileReader::ReadRaw(void * /*aData*/, size_t /*aSizeInBytes*/, size_t /*aOffset*/)
{
}
FileType PseudoFileReader::GetFileType() const
{
    return FileType::UNKNOWN;
}
} // namespace mpp::fileIO