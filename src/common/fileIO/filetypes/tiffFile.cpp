#include "tiffFile.h"
#include "bigTiffImageFileDirectory.h"
#include "tiffImageFileDirectory.h"
#include <algorithm>
#include <array>
#include <common/defines.h>
#include <common/exception.h>
#include <common/fileIO/file.h>
#include <common/fileIO/fileIOException.h>
#include <common/fileIO/fileReader.h>
#include <common/fileIO/lzwCompression.h>
#include <common/fileIO/zlibCompression.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/safeCast.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

namespace opp::fileIO
{
using namespace tiffTag;
using namespace bigTiffTag;

struct membuf : std::streambuf
{
    membuf(char *aBegin, char *aEnd)
    {
        this->setg(aBegin, aBegin, aEnd);
    }
};

TIFFFile::TIFFFile(const std::filesystem::path &aFileName) : File(aFileName)
{
}

TIFFFile::TIFFFile(const opp::image::Size2D &aSize, opp::image::PixelTypeEnum aDataType, double aPixelSize) : File("")
{
    if (!CanWriteAs(aSize, aDataType))
    {
        throw INVALIDARGUMENT(aSize / aDataType, "A TIFFFile with dimensions " << aSize << " and datatype " << aDataType
                                                                               << " is not supported.");
    }
    TIFFFile::SetDataType(aDataType);
    mWidth  = aSize.x;
    mHeight = aSize.y;
    mPlanes = 1;

    Rational pixelSize(720, 10); // NOLINT
    if (aPixelSize != 0)
    {
        pixelSize = TIFFConvertPixelSizeToDPI(aPixelSize);
    }
    mPixelSize = TIFFConvertDPIToPixelSize(pixelSize);

    mImageFileDirectories.clear();

    mImageFileDirectories.emplace_back(std::make_shared<ImageFileDirectory>(
        to_uint(mWidth), to_uint(mHeight), aPixelSize, to_ushort(mBitsPerSample), to_ushort(mSamplesPerPixel),
        mSampleFormat, mIsPlanar, mPhotometricInterpretation));

    mData.resize(TIFFFile::DataSize(), 0);
}

void TIFFFile::OpenAndRead()
{
    OpenFileForReading();
    SetIsLittleEndian(true);
    bool needEndianessInverse = false;

    Read(reinterpret_cast<char *>(&mTiffFileHeader), sizeof(mTiffFileHeader));

    if (mTiffFileHeader.BytaOrder == 0x4949) // NOLINT little Endian
    {
        needEndianessInverse = false; // We are on little Endian hardware...
    }
    else if (mTiffFileHeader.BytaOrder == 0x4D4D) // NOLINT big Endian
    {
        needEndianessInverse = true;
    }
    else
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file.");
    }

    if (needEndianessInverse)
    {
        EndianSwap(mTiffFileHeader.ID);
        EndianSwap(mTiffFileHeader.OffsetToIFD);
        SetIsLittleEndian(false);
    }

    if (mTiffFileHeader.ID == 42) // NOLINT good old TIFF
    {
        ReadTIFF();
    }
    else
    {
        if (mTiffFileHeader.ID == 43) // NOLINT bigTIFF
        {
            // restart and try BigTiff
            SeekRead(0);

            Read(reinterpret_cast<char *>(&mBigTiffFileHeader), sizeof(mBigTiffFileHeader));

            if (mBigTiffFileHeader.BytaOrder == 0x4949) // NOLINT little Endian
            {
                needEndianessInverse = false; // We are on ittle Endian hardware...
            }
            else if (mBigTiffFileHeader.BytaOrder == 0x4D4D) // NOLINT big Endian
            {
                needEndianessInverse = true;
            }
            else
            {
                CloseFileForReading();
                throw FILEIOEXCEPTION(FileName(),
                                      "This doesn't seem to be a valid BigTIFF file: Error in file header.");
            }

            if (needEndianessInverse)
            {
                EndianSwap(mBigTiffFileHeader.ID);
                EndianSwap(mBigTiffFileHeader.OffsetToIFD);
                SetIsLittleEndian(false);
            }

            if (mBigTiffFileHeader.MustBeZero != 0)
            {
                CloseFileForReading();
                throw FILEIOEXCEPTION(FileName(),
                                      "This doesn't seem to be a valid BigTIFF file: Error in file header.");
            }

            if (mBigTiffFileHeader.OffsetByteSize != 8)
            {
                CloseFileForReading();
                throw FILEIOEXCEPTION(FileName(),
                                      "This doesn't seem to be a valid BigTIFF file: Error in file header.");
            }

            ReadBigTIFF();
        }
        else
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file.");
        }
    }

    mData.resize(DataSize(), 0);

    // read all planes:
    for (size_t plane = 0; plane < to_size_t(mPlanes); plane++)
    {
        if (ReadStatusCallback())
        {
            const FileReader::Status status{to_size_t(mPlanes), plane};
            ReadStatusCallback()(status);
        }
        ReadPlane(plane);
    }
    if (ReadStatusCallback())
    {
        const FileReader::Status status{to_size_t(mPlanes), to_size_t(mPlanes)};
        ReadStatusCallback()(status);
    }

    const bool ok = ReadStream()->good();
    CloseFileForReading();

    if (!ok)
    {
        throw FILEIOEXCEPTION(
            FileName(), "Error while reading from file stream. Something must have been wrong before this point.");
    }
}

void TIFFFile::OpenAndReadHeader()
{
    OpenFileForReading();

    bool needEndianessInverse = false;

    Read(reinterpret_cast<char *>(&mTiffFileHeader), sizeof(mTiffFileHeader));

    if (mTiffFileHeader.BytaOrder == 0x4949) // NOLINT little Endian
    {
        needEndianessInverse = false; // We are on ittle Endian hardware...
    }
    else if (mTiffFileHeader.BytaOrder == 0x4D4D) // NOLINT big Endian
    {
        needEndianessInverse = true;
    }
    else
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file.");
    }

    if (needEndianessInverse)
    {
        EndianSwap(mTiffFileHeader.ID);
        EndianSwap(mTiffFileHeader.OffsetToIFD);
        SetIsLittleEndian(false);
    }

    if (mTiffFileHeader.ID == 42) // NOLINT good old TIFF
    {
        ReadTIFF();
    }
    else
    {
        if (mTiffFileHeader.ID == 43) // NOLINT bigTIFF
        {
            // restart and try BigTiff
            SeekRead(0);

            Read(reinterpret_cast<char *>(&mBigTiffFileHeader), sizeof(mBigTiffFileHeader));

            if (mBigTiffFileHeader.BytaOrder == 0x4949) // NOLINT little Endian
            {
                needEndianessInverse = false; // We are on little Endian hardware...
            }
            else if (mBigTiffFileHeader.BytaOrder == 0x4D4D) // NOLINT big Endian
            {
                needEndianessInverse = true;
            }
            else
            {
                CloseFileForReading();
                throw FILEIOEXCEPTION(FileName(),
                                      "This doesn't seem to be a valid BigTIFF file: Error in file header.");
            }

            if (needEndianessInverse)
            {
                EndianSwap(mBigTiffFileHeader.ID);
                EndianSwap(mBigTiffFileHeader.OffsetToIFD);
                SetIsLittleEndian(false);
            }

            if (mBigTiffFileHeader.MustBeZero != 0)
            {
                CloseFileForReading();
                throw FILEIOEXCEPTION(FileName(),
                                      "This doesn't seem to be a valid BigTIFF file: Error in file header.");
            }

            if (mBigTiffFileHeader.OffsetByteSize != 8)
            {
                CloseFileForReading();
                throw FILEIOEXCEPTION(FileName(),
                                      "This doesn't seem to be a valid BigTIFF file: Error in file header.");
            }

            ReadBigTIFF();
        }
        else
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file.");
        }
    }

    const bool ok = ReadStream()->good();
    CloseFileForReading();

    if (!ok)
    {
        throw FILEIOEXCEPTION(
            FileName(), "Error while reading from file stream. Something must have been wrong before this point.");
    }
}

bool TIFFFile::TryToOpenAndReadHeader() noexcept
{
    try
    {
        bool ok = TryToOpenFileForReading();
        if (!ok)
        {
            return false;
        }

        bool needEndianessInverse = false;

        Read(reinterpret_cast<char *>(&mTiffFileHeader), sizeof(mTiffFileHeader));

        if (mTiffFileHeader.BytaOrder == 0x4949) // NOLINT little Endian
        {
            needEndianessInverse = false; // We are on ittle Endian hardware...
        }
        else if (mTiffFileHeader.BytaOrder == 0x4D4D) // NOLINT big Endian
        {
            needEndianessInverse = true;
        }
        else
        {
            CloseFileForReading();
            return false;
        }

        if (needEndianessInverse)
        {
            EndianSwap(mTiffFileHeader.ID);
            EndianSwap(mTiffFileHeader.OffsetToIFD);
            SetIsLittleEndian(false);
        }

        if (mTiffFileHeader.ID == 42) // NOLINT good old TIFF
        {
            ReadTIFF();
        }
        else
        {
            if (mTiffFileHeader.ID == 43) // NOLINT bigTIFF
            {
                // restart and try BigTiff
                SeekRead(0);

                Read(reinterpret_cast<char *>(&mBigTiffFileHeader), sizeof(mBigTiffFileHeader));

                if (mBigTiffFileHeader.BytaOrder == 0x4949) // NOLINT little Endian
                {
                    needEndianessInverse = false; // We are on ittle Endian hardware...
                }
                else if (mBigTiffFileHeader.BytaOrder == 0x4D4D) // NOLINT big Endian
                {
                    needEndianessInverse = true;
                }
                else
                {
                    CloseFileForReading();
                    return false;
                }

                if (needEndianessInverse)
                {
                    EndianSwap(mBigTiffFileHeader.ID);
                    EndianSwap(mBigTiffFileHeader.OffsetToIFD);
                    SetIsLittleEndian(false);
                }

                if (mBigTiffFileHeader.MustBeZero != 0)
                {
                    CloseFileForReading();
                    return false;
                }

                if (mBigTiffFileHeader.OffsetByteSize != 8)
                {
                    CloseFileForReading();
                    return false;
                }

                ReadBigTIFF();
            }
            else
            {
                CloseFileForReading();
                return false;
            }
        }

        ok = ReadStream()->good();
        CloseFileForReading();

        return ok;
    }
    catch (const std::exception &)
    {
        return false;
    }
}

void TIFFFile::OpenAndWrite()
{
    if (mImageFileDirectories.size() != 1)
    {
        throw FILEIOEXCEPTION(FileName(), "Only standard TIFF files (not BigTIFF) with one image plane without "
                                          "compression are supported for writing.");
    }

    // check if IFD is compatible:
    const std::shared_ptr<ImageFileDirectory> ifd = mImageFileDirectories[0];

    bool haveIFDImageWidth                = false;
    bool haveIFDImageLength               = false;
    bool haveIFDBitsPerSample             = false;
    bool haveIFDSampleFormat              = false;
    bool haveIFDCompression               = false;
    bool haveIFDPhotometricInterpretation = false;
    bool haveIFDStripOffsets              = false;
    bool haveIFDSamplesPerPixel           = false;
    bool haveIFDRowsPerStrip              = false;
    bool haveIFDStripByteCounts           = false;
    bool haveIFDXResolution               = false;
    bool haveIFDYResolution               = false;
    bool haveIFDResolutionUnit            = false;
    bool haveIFDSoftware                  = false;

    for (auto &elem : ifd->GetEntries())
    {
        if (elem->GetTagID() == IFDImageWidth::TagID)
        {
            haveIFDImageWidth = true;
        }
        if (elem->GetTagID() == IFDImageLength::TagID)
        {
            haveIFDImageLength = true;
        }
        if (elem->GetTagID() == IFDBitsPerSample::TagID)
        {
            haveIFDBitsPerSample = true;
        }
        if (elem->GetTagID() == IFDSampleFormat::TagID)
        {
            haveIFDSampleFormat = true;
        }
        if (elem->GetTagID() == IFDCompression::TagID)
        {
            haveIFDCompression = true;
        }
        if (elem->GetTagID() == IFDPhotometricInterpretation::TagID)
        {
            haveIFDPhotometricInterpretation = true;
        }
        if (elem->GetTagID() == IFDStripOffsets::TagID)
        {
            haveIFDStripOffsets = true;
        }
        if (elem->GetTagID() == IFDSamplesPerPixel::TagID)
        {
            haveIFDSamplesPerPixel = true;
        }
        if (elem->GetTagID() == IFDRowsPerStrip::TagID)
        {
            haveIFDRowsPerStrip = true;
        }
        if (elem->GetTagID() == IFDStripByteCounts::TagID)
        {
            haveIFDStripByteCounts = true;
        }
        if (elem->GetTagID() == IFDXResolution::TagID)
        {
            haveIFDXResolution = true;
        }
        if (elem->GetTagID() == IFDYResolution::TagID)
        {
            haveIFDYResolution = true;
        }
        if (elem->GetTagID() == IFDResolutionUnit::TagID)
        {
            haveIFDResolutionUnit = true;
        }
        if (elem->GetTagID() == IFDSoftware::TagID)
        {
            haveIFDSoftware = true;
        }
    }

    if (!haveIFDImageWidth)
    {
        throw FILEIOEXCEPTION(FileName(), "Mandatory TIFF tag '" << IFDImageWidth::TagName << "' is missing.");
    }
    if (!haveIFDImageLength)
    {
        throw FILEIOEXCEPTION(FileName(), "Mandatory TIFF tag '" << IFDImageLength::TagName << "' is missing.");
    }
    if (!haveIFDBitsPerSample)
    {
        throw FILEIOEXCEPTION(FileName(), "Mandatory TIFF tag '" << IFDBitsPerSample::TagName << "' is missing.");
    }
    if (!haveIFDSampleFormat)
    {
        throw FILEIOEXCEPTION(FileName(), "Mandatory TIFF tag '" << IFDSampleFormat::TagName << "' is missing.");
    }
    if (!haveIFDCompression)
    {
        throw FILEIOEXCEPTION(FileName(), "Mandatory TIFF tag '" << IFDCompression::TagName << "' is missing.");
    }
    if (!haveIFDPhotometricInterpretation)
    {
        throw FILEIOEXCEPTION(FileName(),
                              "Mandatory TIFF tag '" << IFDPhotometricInterpretation::TagName << "' is missing.");
    }
    if (!haveIFDStripOffsets)
    {
        throw FILEIOEXCEPTION(FileName(), "Mandatory TIFF tag '" << IFDStripOffsets::TagName << "' is missing.");
    }
    if (!haveIFDSamplesPerPixel)
    {
        throw FILEIOEXCEPTION(FileName(), "Mandatory TIFF tag '" << IFDSamplesPerPixel::TagName << "' is missing.");
    }
    if (!haveIFDRowsPerStrip)
    {
        throw FILEIOEXCEPTION(FileName(), "Mandatory TIFF tag '" << IFDRowsPerStrip::TagName << "' is missing.");
    }
    if (!haveIFDStripByteCounts)
    {
        throw FILEIOEXCEPTION(FileName(), "Mandatory TIFF tag '" << IFDStripByteCounts::TagName << "' is missing.");
    }
    if (!haveIFDXResolution)
    {
        throw FILEIOEXCEPTION(FileName(), "Mandatory TIFF tag '" << IFDXResolution::TagName << "' is missing.");
    }
    if (!haveIFDYResolution)
    {
        throw FILEIOEXCEPTION(FileName(), "Mandatory TIFF tag '" << IFDYResolution::TagName << "' is missing.");
    }
    if (!haveIFDResolutionUnit)
    {
        throw FILEIOEXCEPTION(FileName(), "Mandatory TIFF tag '" << IFDResolutionUnit::TagName << "' is missing.");
    }
    if (!haveIFDSoftware)
    {
        throw FILEIOEXCEPTION(FileName(), "Mandatory TIFF tag '" << IFDSoftware::TagName << "' is missing.");
    }

    // check values:
    const std::shared_ptr<IFDCompression> compression =
        std::dynamic_pointer_cast<IFDCompression>(ifd->GetEntry(IFDCompression::TagID));
    if (compression->Value() != TIFFCompression::NoCompression)
    {
        throw FILEIOEXCEPTION(FileName(), "Only standard TIFF files without compression are supported for writing.");
    }

    const std::shared_ptr<IFDStripOffsets> stripOffsets =
        std::dynamic_pointer_cast<IFDStripOffsets>(ifd->GetEntry(IFDStripOffsets::TagID));
    if (stripOffsets->Value().size() != 1)
    {
        throw FILEIOEXCEPTION(FileName(), "Only TIFF files with one strip offset IFD is supported.");
    }

    const std::shared_ptr<IFDRowsPerStrip> rowsPerStrip =
        std::dynamic_pointer_cast<IFDRowsPerStrip>(ifd->GetEntry(IFDRowsPerStrip::TagID));
    if (rowsPerStrip->Value() != to_uint(mHeight))
    {
        throw FILEIOEXCEPTION(
            FileName(),
            "Only TIFF files with one strip offset IFD where the strip has image height rows is supported.");
    }

    const std::shared_ptr<IFDStripByteCounts> stripByteCount =
        std::dynamic_pointer_cast<IFDStripByteCounts>(ifd->GetEntry(IFDStripByteCounts::TagID));
    if (stripByteCount->Value().size() != 1)
    {
        throw FILEIOEXCEPTION(FileName(), "Only TIFF files with one strip byte count IFD is supported.");
    }
    if (stripByteCount->Value()[0] != DataSize())
    {
        throw FILEIOEXCEPTION(FileName(), "Strip byte count is not the same as the data size. StripByteCount: "
                                              << stripByteCount->Value()[0] << " data size: " << DataSize() << ".");
    }

    // Now that we are sure we can write the data:
    OpenFileForWriting(FileOpenMode::EraseOldFile);

    std::array<char, 8> header{0x49, 0x49, 0x2A, 00, 8, 0, 0, 0}; // NOLINT Tiff header with offset to first IFD=8

    Write(header.data(), header.size());

    const std::shared_ptr<ImageFileDirectory> dir = mImageFileDirectories[0];

    ushort entryCount = to_ushort(dir->GetEntries().size());

    Write(reinterpret_cast<char *>(&entryCount), sizeof(ushort));

    for (auto &entry : dir->GetEntries())
    {
        entry->SavePass1(*GetWriteStream());
    }

    uint newIfd = 0; // marker that no more IFDs are coming
    Write(reinterpret_cast<char *>(&newIfd), 4);

    for (auto &entry : dir->GetEntries())
    {
        entry->SavePass2(*GetWriteStream());
    }

    SeekWrite(0, std::ios_base::end);

    size_t finalImageOffset = TellWrite();
    finalImageOffset += finalImageOffset % 4;
    std::vector<uint> finallOffsets{to_uint(finalImageOffset)};
    std::dynamic_pointer_cast<IFDStripOffsets>(dir->GetEntry(IFDStripOffsets::TagID))
        ->SaveFinalOffsets(*GetWriteStream(), finallOffsets);

    SeekWrite(finalImageOffset, std::ios_base::beg);

    // write finally the image data:
    Write(reinterpret_cast<const char *>(mData.data()), DataSize());

    const bool ok = GetWriteStream()->good();

    CloseFileForWriting();

    if (!ok)
    {
        throw FILEIOEXCEPTION(FileName(),
                              "Error while writing to file stream. Something must have been wrong before this point.");
    }
}

void TIFFFile::SaveAs(const std::filesystem::path &aFileName)
{
    if (!CanWriteAs(Vec3i(mWidth, mHeight, mPlanes), GetDataType()))
    {

        throw FILEIOEXCEPTION(FileName(), "A TIFF file with datatype '" << GetDataType() << "' and dimensions "
                                                                        << mWidth << " x " << mHeight << " x "
                                                                        << mPlanes << " is not supported for writing.");
    }
    SetIsLittleEndian(true);
    SetFileName(aFileName);

    // change tags to what we can write:
    mImageFileDirectories.clear();
    mBigImageFileDirectories.clear();
    mImageFileDirectories.push_back(std::make_shared<ImageFileDirectory>(
        to_uint(mWidth), to_uint(mHeight), mPixelSize, to_ushort(mBitsPerSample), to_ushort(mSamplesPerPixel),
        mSampleFormat, mIsPlanar, mPhotometricInterpretation));

    OpenAndWrite();
}

void TIFFFile::ReadTIFF()
{
    const bool ok = ReadStream()->good();

    if (!ok)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(
            FileName(), "Error while reading from file stream. Something must have been wrong before this point.");
    }

    SeekRead(mTiffFileHeader.OffsetToIFD, std::ios_base::beg);

    while (true)
    {
        const std::shared_ptr<ImageFileDirectory> ifd = std::make_shared<ImageFileDirectory>(*this);
        mImageFileDirectories.push_back(ifd);

        const uint offsetToNext = ReadLE<uint>();
        if (offsetToNext == 0)
        {
            break;
        }
        SeekRead(offsetToNext, std::ios_base::beg);
    }

    if (!ReadStream()->good())
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(
            FileName(), "Error while reading from file stream. Something must have been wrong before this point.");
    }

    // read the mandatory tags:
    const std::shared_ptr<ImageFileDirectory> first = mImageFileDirectories[0];

    const std::shared_ptr<IFDCompression> compressionIFD =
        std::dynamic_pointer_cast<IFDCompression>(first->GetEntry(IFDCompression::TagID));
    if (!compressionIFD)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '" << IFDCompression::TagName
                                                                                             << "' is missing");
    }
    mCompression = compressionIFD->Value();

    const std::shared_ptr<IFDImageWidth> widthIFD =
        std::dynamic_pointer_cast<IFDImageWidth>(first->GetEntry(IFDImageWidth::TagID));
    if (!widthIFD)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '" << IFDImageWidth::TagName
                                                                                             << "' is missing");
    }
    mWidth = to_int(widthIFD->Value());

    const std::shared_ptr<IFDImageLength> heightIFD =
        std::dynamic_pointer_cast<IFDImageLength>(first->GetEntry(IFDImageLength::TagID));
    if (!heightIFD)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '" << IFDImageLength::TagName
                                                                                             << "' is missing");
    }
    mHeight = to_int(heightIFD->Value());

    const std::shared_ptr<IFDSamplesPerPixel> SPPIFD =
        std::dynamic_pointer_cast<IFDSamplesPerPixel>(first->GetEntry(IFDSamplesPerPixel::TagID));
    if (!SPPIFD)
    {
        mSamplesPerPixel = 1;
    }
    else
    {
        mSamplesPerPixel = SPPIFD->Value();
    }

    const std::shared_ptr<IFDBitsPerSample> BPSIFD =
        std::dynamic_pointer_cast<IFDBitsPerSample>(first->GetEntry(IFDBitsPerSample::TagID));
    if (!BPSIFD)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                              << IFDBitsPerSample::TagName << "' is missing");
    }
    mBitsPerSample          = BPSIFD->Value(0);
    size_t pixelSizeInBytes = 0;
    for (size_t i = 0; i < to_size_t(mSamplesPerPixel); i++)
    {
        const ushort sampleSize = BPSIFD->Value(i);
        if (sampleSize % 8 != 0)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "Cannot read TIFF files with not byte aligned pixel sizes. Got: "
                                                  << sampleSize << " bits per sample.");
        }
        pixelSizeInBytes += sampleSize / 8;
    }

    const std::shared_ptr<IFDSampleFormat> sampleFormatIFD =
        std::dynamic_pointer_cast<IFDSampleFormat>(first->GetEntry(IFDSampleFormat::TagID));
    if (sampleFormatIFD)
    {
        mSampleFormat = sampleFormatIFD->Value();
    }

    const std::shared_ptr<IFDPlanarConfiguration> planarIFD =
        std::dynamic_pointer_cast<IFDPlanarConfiguration>(first->GetEntry(IFDPlanarConfiguration::TagID));
    if (planarIFD)
    {
        mIsPlanar = planarIFD->Value() == TIFFPlanarConfigurartion::Planar;
    }

    const std::shared_ptr<IFDDifferencingPredictor> differentialIFD =
        std::dynamic_pointer_cast<IFDDifferencingPredictor>(first->GetEntry(IFDDifferencingPredictor::TagID));
    if (differentialIFD)
    {
        mDifferencingPredictor = differentialIFD->Value();
    }

    const std::shared_ptr<IFDOrientation> orientationIFD =
        std::dynamic_pointer_cast<IFDOrientation>(first->GetEntry(IFDOrientation::TagID));
    if (orientationIFD)
    {
        mOrientation = orientationIFD->Value();
    }

    const std::shared_ptr<IFDXResolution> ifdResX =
        std::dynamic_pointer_cast<IFDXResolution>(first->GetEntry(IFDXResolution::TagID));
    if (ifdResX)
    {
        // SerialEM stores pixel size as DPI -> convert inch to cm to nm
        mPixelSize = TIFFConvertDPIToPixelSize(ifdResX->Value());
    }

    mPlanes = to_int(mImageFileDirectories.size());

    size_t ifdCounter = 0;
    // check if all IFDs have same image size
    for (auto &ifd : mImageFileDirectories)
    {
        const std::shared_ptr<IFDImageWidth> width =
            std::dynamic_pointer_cast<IFDImageWidth>(ifd->GetEntry(IFDImageWidth::TagID));
        if (!width)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                                  << IFDImageWidth::TagName << "' is missing in IFD: " << ifdCounter);
        }
        const std::shared_ptr<IFDImageLength> height =
            std::dynamic_pointer_cast<IFDImageLength>(ifd->GetEntry(IFDImageLength::TagID));
        if (!height)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                                  << IFDImageLength::TagName << "' is missing in IFD: " << ifdCounter);
        }
        const std::shared_ptr<IFDBitsPerSample> sampleSize =
            std::dynamic_pointer_cast<IFDBitsPerSample>(ifd->GetEntry(IFDBitsPerSample::TagID));
        if (!sampleSize)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                                  << IFDBitsPerSample::TagName
                                                  << "' is missing in IFD: " << ifdCounter);
        }
        const std::shared_ptr<IFDCompression> compression =
            std::dynamic_pointer_cast<IFDCompression>(ifd->GetEntry(IFDCompression::TagID));
        if (!compression)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                                  << IFDCompression::TagName << "' missing in IFD: " << ifdCounter);
        }
        if (to_int(width->Value()) != mWidth || to_int(height->Value()) != mHeight ||
            sampleSize->Value(0) != mBitsPerSample || compression->Value() != mCompression)
        {
            // not all images have the same size / type, this is hence not an image stack
            mPlanes = 1;
            break;
        }
        ifdCounter++;
    }

    if (mPlanes <= 0)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(), "No image IFDs found in file.");
    }

    if (mPlanes > 1 && mIsPlanar)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(), "Only 2D images (no multiple IFDs) are supported for planar images.");
    }

    mReadSegments.resize(to_size_t(mPlanes));
    // store the offset information in internal structure for later reading:
    for (size_t plane = 0; plane < to_size_t(mPlanes); plane++)
    {
        const std::shared_ptr<IFDStripOffsets> offsetIFD =
            std::dynamic_pointer_cast<IFDStripOffsets>(mImageFileDirectories[plane]->GetEntry(IFDStripOffsets::TagID));
        if (!offsetIFD)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                                  << IFDStripOffsets::TagName << "' missing in IFD: " << plane);
        }

        const std::shared_ptr<IFDStripByteCounts> SBCIFD = std::dynamic_pointer_cast<IFDStripByteCounts>(
            mImageFileDirectories[plane]->GetEntry(IFDStripByteCounts::TagID));
        if (!SBCIFD)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                                  << IFDStripByteCounts::TagName << "' missing in IFD: " << plane);
        }

        const std::shared_ptr<IFDRowsPerStrip> ifdRowsPerStrip =
            std::dynamic_pointer_cast<IFDRowsPerStrip>(mImageFileDirectories[plane]->GetEntry(IFDRowsPerStrip::TagID));
        if (!ifdRowsPerStrip)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                                  << IFDRowsPerStrip::TagName << "' missing in IFD: " << plane);
        }
        const size_t rowsPerStrip = ifdRowsPerStrip->Value();

        // check consistency:
        const size_t stripeCount = SBCIFD->Value().size();

        if (offsetIFD->Value().size() != SBCIFD->Value().size())
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(
                FileName(), "Not the same number of byte count and offset fields: This is not a correct TIFF file.");
        }

        size_t offsetInImage = 0;
        mReadSegments[plane].resize(stripeCount);

        if (mIsPlanar)
        {
            // note: only one image plane/slice (=2D image) is supported for planar images.
            size_t colorPlane = 0;
            for (size_t stripe = 0; stripe < stripeCount; stripe++)
            {
                const size_t offsetInFile = offsetIFD->Value()[stripe];
                const size_t toRead       = SBCIFD->Value()[stripe];
                const size_t destOffset   = offsetInImage;

                mReadSegments[plane][stripe].DataOffset        = offsetInFile;
                mReadSegments[plane][stripe].DataSize          = toRead;
                mReadSegments[plane][stripe].DestinationOffset = destOffset;
                mReadSegments[plane][stripe].RowsPerStrip      = rowsPerStrip;

                offsetInImage += rowsPerStrip * to_size_t(mWidth) * to_size_t(mBitsPerSample) / 8;

                // handle the case when a strip goes beyond one plane of a color channel:
                if (offsetInImage >= (1 + colorPlane) * GetImageSizeInBytes())
                {
                    colorPlane++;
                    offsetInImage = colorPlane * GetImageSizeInBytes();
                }
            }
        }
        else
        {
            for (size_t stripe = 0; stripe < stripeCount; stripe++)
            {
                const size_t offsetInFile = offsetIFD->Value()[stripe];
                const size_t toRead       = SBCIFD->Value()[stripe];
                const size_t destOffset   = plane * GetImageSizeInBytes() + offsetInImage;

                mReadSegments[plane][stripe].DataOffset        = offsetInFile;
                mReadSegments[plane][stripe].DataSize          = toRead;
                mReadSegments[plane][stripe].DestinationOffset = destOffset;
                mReadSegments[plane][stripe].RowsPerStrip      = rowsPerStrip;

                offsetInImage += rowsPerStrip * to_size_t(mWidth) * pixelSizeInBytes;
            }
        }
    }
}

void TIFFFile::ReadBigTIFF()
{
    const bool ok = ReadStream()->good();

    if (!ok)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(
            FileName(), "Error while reading from file stream. Something must have been wrong before this point.");
    }

    SeekRead(mBigTiffFileHeader.OffsetToIFD, std::ios_base::beg);

    while (true)
    {
        const std::shared_ptr<BigImageFileDirectory> ifd = std::make_shared<BigImageFileDirectory>(*this);
        mBigImageFileDirectories.push_back(ifd);

        const ulong64 offsetToNext = ReadLE<ulong64>();
        if (offsetToNext == 0)
        {
            break;
        }
        SeekRead(offsetToNext, std::ios_base::beg);
    }

    // read the mandatory tags:
    const std::shared_ptr<BigImageFileDirectory> first = mBigImageFileDirectories[0];

    const std::shared_ptr<BigIFDCompression> compressionIFD =
        std::dynamic_pointer_cast<BigIFDCompression>(first->GetEntry(BigIFDCompression::TagID));
    if (!compressionIFD)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                              << BigIFDCompression::TagName << "' is missing");
    }
    mCompression = compressionIFD->Value();

    const std::shared_ptr<BigIFDImageWidth> widthIFD =
        std::dynamic_pointer_cast<BigIFDImageWidth>(first->GetEntry(BigIFDImageWidth::TagID));
    if (!widthIFD)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                              << BigIFDImageWidth::TagName << "' is missing");
    }
    mWidth = to_int(widthIFD->Value());

    const std::shared_ptr<BigIFDImageLength> heightIFD =
        std::dynamic_pointer_cast<BigIFDImageLength>(first->GetEntry(BigIFDImageLength::TagID));
    if (!heightIFD)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                              << BigIFDImageLength::TagName << "' is missing");
    }
    mHeight = to_int(heightIFD->Value());

    const std::shared_ptr<BigIFDSamplesPerPixel> SPPIFD =
        std::dynamic_pointer_cast<BigIFDSamplesPerPixel>(first->GetEntry(BigIFDSamplesPerPixel::TagID));
    if (!SPPIFD)
    {
        mSamplesPerPixel = 1;
    }
    else
    {
        mSamplesPerPixel = SPPIFD->Value();
    }

    const std::shared_ptr<BigIFDBitsPerSample> BPSIFD =
        std::dynamic_pointer_cast<BigIFDBitsPerSample>(first->GetEntry(BigIFDBitsPerSample::TagID));
    if (!BPSIFD)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                              << BigIFDBitsPerSample::TagName << "' is missing");
    }
    mBitsPerSample          = BPSIFD->Value(0);
    size_t pixelSizeInBytes = 0;
    for (size_t i = 0; i < to_size_t(mSamplesPerPixel); i++)
    {
        const ushort sampleSize = BPSIFD->Value(i);
        if (sampleSize % 8 != 0)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "Cannot read TIFF files with not byte aligned pixel sizes. Got: "
                                                  << sampleSize << " bits per sample.");
        }
        pixelSizeInBytes += sampleSize / 8;
    }

    const std::shared_ptr<BigIFDSampleFormat> sampleFormatIFD =
        std::dynamic_pointer_cast<BigIFDSampleFormat>(first->GetEntry(BigIFDSampleFormat::TagID));
    if (sampleFormatIFD)
    {
        mSampleFormat = sampleFormatIFD->Value();
    }

    const std::shared_ptr<BigIFDPlanarConfiguration> planarIFD =
        std::dynamic_pointer_cast<BigIFDPlanarConfiguration>(first->GetEntry(BigIFDPlanarConfiguration::TagID));
    if (planarIFD)
    {
        mIsPlanar = planarIFD->Value() == TIFFPlanarConfigurartion::Planar;
    }

    const std::shared_ptr<BigIFDDifferencingPredictor> differentialIFD =
        std::dynamic_pointer_cast<BigIFDDifferencingPredictor>(first->GetEntry(BigIFDDifferencingPredictor::TagID));
    if (differentialIFD)
    {
        mDifferencingPredictor = differentialIFD->Value();
    }

    const std::shared_ptr<BigIFDOrientation> orientationIFD =
        std::dynamic_pointer_cast<BigIFDOrientation>(first->GetEntry(BigIFDOrientation::TagID));
    if (orientationIFD)
    {
        mOrientation = orientationIFD->Value();
    }

    const std::shared_ptr<BigIFDXResolution> ifdResX =
        std::dynamic_pointer_cast<BigIFDXResolution>(first->GetEntry(BigIFDXResolution::TagID));
    if (ifdResX)
    {
        mPixelSize = TIFFConvertDPIToPixelSize(ifdResX->Value());
    }

    mPlanes = to_int(mBigImageFileDirectories.size());

    size_t ifdCounter = 0;
    // check if all IFDs have same image size
    for (auto &ifd : mBigImageFileDirectories)
    {
        const std::shared_ptr<BigIFDImageWidth> width =
            std::dynamic_pointer_cast<BigIFDImageWidth>(ifd->GetEntry(BigIFDImageWidth::TagID));
        if (!width)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                                  << IFDImageWidth::TagName << "' is missing in IFD: " << ifdCounter);
        }
        const std::shared_ptr<BigIFDImageLength> height =
            std::dynamic_pointer_cast<BigIFDImageLength>(ifd->GetEntry(BigIFDImageLength::TagID));
        if (!height)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                                  << IFDImageLength::TagName << "' is missing in IFD: " << ifdCounter);
        }
        const std::shared_ptr<BigIFDBitsPerSample> sampleSize =
            std::dynamic_pointer_cast<BigIFDBitsPerSample>(ifd->GetEntry(BigIFDBitsPerSample::TagID));
        if (!sampleSize)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                                  << IFDBitsPerSample::TagName
                                                  << "' is missing in IFD: " << ifdCounter);
        }
        const std::shared_ptr<BigIFDCompression> compression =
            std::dynamic_pointer_cast<BigIFDCompression>(ifd->GetEntry(BigIFDCompression::TagID));
        if (!compression)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                                  << IFDCompression::TagName << "' missing in IFD: " << ifdCounter);
        }
        if (to_int(width->Value()) != mWidth || to_int(height->Value()) != mHeight ||
            sampleSize->Value(0) != mBitsPerSample || compression->Value() != mCompression)
        {
            // not all images have the same size / type, this is hence not an image stack
            mPlanes = 1;
            break;
        }
        ifdCounter++;
    }

    if (mPlanes <= 0)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(), "No image IFDs found in file.");
    }

    if (mPlanes > 1 && mIsPlanar)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(), "Only 2D images (no multiple IFDs) are supported for planar images.");
    }

    mReadSegments.resize(to_size_t(mPlanes));
    // store the offset information in internal structure for later reading:
    for (size_t plane = 0; plane < to_size_t(mPlanes); plane++)
    {
        const std::shared_ptr<BigIFDStripOffsets> offsetIFD = std::dynamic_pointer_cast<BigIFDStripOffsets>(
            mBigImageFileDirectories[plane]->GetEntry(BigIFDStripOffsets::TagID));
        if (!offsetIFD)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                                  << IFDStripOffsets::TagName << "' missing in IFD: " << plane);
        }

        const std::shared_ptr<BigIFDStripByteCounts> SBCIFD = std::dynamic_pointer_cast<BigIFDStripByteCounts>(
            mBigImageFileDirectories[plane]->GetEntry(BigIFDStripByteCounts::TagID));
        if (!SBCIFD)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                                  << IFDStripByteCounts::TagName << "' missing in IFD: " << plane);
        }

        const std::shared_ptr<BigIFDRowsPerStrip> ifdRowsPerStrip = std::dynamic_pointer_cast<BigIFDRowsPerStrip>(
            mBigImageFileDirectories[plane]->GetEntry(BigIFDRowsPerStrip::TagID));
        if (!ifdRowsPerStrip)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "This doesn't seem to be a valid TIFF file: Tag '"
                                                  << IFDRowsPerStrip::TagName << "' missing in IFD: " << plane);
        }
        const size_t rowsPerStrip = ifdRowsPerStrip->Value();

        // check consistency:
        const size_t stripeCount = SBCIFD->Value().size();

        if (offsetIFD->Value().size() != SBCIFD->Value().size())
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(
                FileName(), "Not the same number of byte count and offset fields: This is not a correct TIFF file.");
        }

        size_t offsetInImage = 0;
        mReadSegments[plane].resize(stripeCount);

        if (mIsPlanar)
        {
            // note: only one image plane/slice (=2D image) is supported for planar images.
            size_t colorPlane = 0;
            for (size_t stripe = 0; stripe < stripeCount; stripe++)
            {
                const size_t offsetInFile = offsetIFD->Value()[stripe];
                const size_t toRead       = SBCIFD->Value()[stripe];
                const size_t destOffset   = offsetInImage;

                mReadSegments[plane][stripe].DataOffset        = offsetInFile;
                mReadSegments[plane][stripe].DataSize          = toRead;
                mReadSegments[plane][stripe].DestinationOffset = destOffset;
                mReadSegments[plane][stripe].RowsPerStrip      = rowsPerStrip;

                offsetInImage += rowsPerStrip * to_size_t(mWidth) * to_size_t(mBitsPerSample) / 8;

                // handle the case when a strip goes beyond one plane of a color channel:
                if (offsetInImage >= (1 + colorPlane) * GetImageSizeInBytes())
                {
                    colorPlane++;
                    offsetInImage = colorPlane * GetImageSizeInBytes();
                }
            }
        }
        else
        {
            for (size_t stripe = 0; stripe < stripeCount; stripe++)
            {
                const size_t offsetInFile = offsetIFD->Value()[stripe];
                const size_t toRead       = SBCIFD->Value()[stripe];
                const size_t destOffset   = plane * GetImageSizeInBytes() + offsetInImage;

                mReadSegments[plane][stripe].DataOffset        = offsetInFile;
                mReadSegments[plane][stripe].DataSize          = toRead;
                mReadSegments[plane][stripe].DestinationOffset = destOffset;
                mReadSegments[plane][stripe].RowsPerStrip      = rowsPerStrip;

                offsetInImage += rowsPerStrip * to_size_t(mWidth) * pixelSizeInBytes;
            }
        }
    }
}

void TIFFFile::ReadPlane(size_t aIdx)
{
    switch (mCompression)
    {
        case TIFFCompression::NoCompression:
            ReadPlaneNoCompression(aIdx);
            DecodeDifferencingPredictor(aIdx);
            return;
        case TIFFCompression::LZW:
            ReadPlaneLZWCompression(aIdx);
            DecodeDifferencingPredictor(aIdx);
            return;
        case TIFFCompression::Deflate:
        case TIFFCompression::DeflateAdobe:
            ReadPlaneDeflateCompression(aIdx);
            DecodeDifferencingPredictor(aIdx);
            return;
        case TIFFCompression::PackBits:
            ReadPlanePackBitsCompression(aIdx);
            DecodeDifferencingPredictor(aIdx);
            return;
        case TIFFCompression::CCITTGroup3:
        case TIFFCompression::EER7Bit:
        case TIFFCompression::EER8Bit:
            break;
    }
    throw FILEIOEXCEPTION(
        FileName(), "Only uncompressed, LZW or Deflate compressed TIFF files are supported. Compression TIFF-tag is: "
                        << int(mCompression));
}

void TIFFFile::DecodeDifferencingPredictor(size_t aIdx)
{
    if (mDifferencingPredictor == TIFFDifferencingPredictor::None)
    {
        return; // Nothing to do here
    }

    // treat void as equivalent as unsigned int as not all TIFF writers set the tag
    if (mSampleFormat != TIFFSampleFormat::UINT && mSampleFormat != TIFFSampleFormat::VOIDTYPE)
    {
        throw FILEIOEXCEPTION(FileName(),
                              "Only TIFF files with unsigned integers are supported for differencing prediction.");
    }

    if (mDifferencingPredictor == TIFFDifferencingPredictor::HorizontalDifferencing)
    {
        const size_t height = to_size_t(Height());
        const size_t width  = to_size_t(Width());
        size_t lineOffset   = 0;

        size_t imageOffset     = aIdx * GetImageSizeInBytes();
        size_t nextPixelOffset = to_size_t(mSamplesPerPixel);
        size_t colorPlanes     = 1;
        size_t planeOffset     = 0;
        if (mIsPlanar)
        {
            imageOffset     = aIdx * GetImageSizeInBytes() * to_size_t(mSamplesPerPixel);
            nextPixelOffset = 1;
            planeOffset     = width * height;
            colorPlanes     = to_size_t(mSamplesPerPixel);
            lineOffset      = width /** to_size_t(mBitsPerSample / 8)*/;
        }
        else
        {
            lineOffset = width * to_size_t(mSamplesPerPixel) /** to_size_t(mBitsPerSample / 8)*/;
        }

        switch (mBitsPerSample)
        {
            case 8:
            {
                for (size_t line = 0; line < height; line++)
                {
                    for (size_t x = 1; x < width; x++)
                    {
                        for (size_t colorPlane = 0; colorPlane < colorPlanes; colorPlane++)
                        {
                            byte *ptr = reinterpret_cast<byte *>(mData.data());
                            byte *thisPixel =
                                ptr + imageOffset + planeOffset * colorPlane + line * lineOffset + x * nextPixelOffset;
                            byte *previousPixel = thisPixel - nextPixelOffset;
                            for (size_t sample = 0; sample < nextPixelOffset; sample++)
                            {
                                *(thisPixel + sample) += *(previousPixel + sample);
                            }
                        }
                    }
                }
                break;
            }
            case 16: // NOLINT
            {
                for (size_t line = 0; line < height; line++)
                {
                    for (size_t x = 1; x < width; x++)
                    {
                        for (size_t colorPlane = 0; colorPlane < colorPlanes; colorPlane++)
                        {
                            ushort *ptr = reinterpret_cast<ushort *>(mData.data());
                            ushort *thisPixel =
                                ptr + imageOffset + planeOffset * colorPlane + line * lineOffset + x * nextPixelOffset;
                            ushort *previousPixel = thisPixel - nextPixelOffset;
                            for (size_t sample = 0; sample < nextPixelOffset; sample++)
                            {
                                *(thisPixel + sample) += *(previousPixel + sample);
                            }
                        }
                    }
                }
                break;
            }
            case 32: // NOLINT
            {
                for (size_t line = 0; line < height; line++)
                {
                    for (size_t x = 1; x < width; x++)
                    {
                        for (size_t colorPlane = 0; colorPlane < colorPlanes; colorPlane++)
                        {
                            uint *ptr = reinterpret_cast<uint *>(mData.data());
                            uint *thisPixel =
                                ptr + imageOffset + planeOffset * colorPlane + line * lineOffset + x * nextPixelOffset;
                            uint *previousPixel = thisPixel - nextPixelOffset;
                            for (size_t sample = 0; sample < nextPixelOffset; sample++)
                            {
                                *(thisPixel + sample) += *(previousPixel + sample);
                            }
                        }
                    }
                }
                break;
            }
            default:
                throw FILEIOEXCEPTION(FileName(), "Only TIFF files with unsigned integers and size per sample of 8, 16 "
                                                  "and 32 bits are supported for differencing prediction.");
                break;
        }
    }
}

void TIFFFile::EncodeDifferencingPredictor(void *aData, opp::image::PixelTypeEnum aDataType, uint aWidth, uint aHeight)
{
    // all unsigned int data types:
    if (aDataType != opp::image::PixelTypeEnum::PTE32uC1 && aDataType != opp::image::PixelTypeEnum::PTE16uC1 &&
        aDataType != opp::image::PixelTypeEnum::PTE8uC1 && aDataType != opp::image::PixelTypeEnum::PTE32uC2 &&
        aDataType != opp::image::PixelTypeEnum::PTE16uC2 && aDataType != opp::image::PixelTypeEnum::PTE8uC2 &&
        aDataType != opp::image::PixelTypeEnum::PTE32uC3 && aDataType != opp::image::PixelTypeEnum::PTE16uC3 &&
        aDataType != opp::image::PixelTypeEnum::PTE8uC3 && aDataType != opp::image::PixelTypeEnum::PTE32uC4 &&
        aDataType != opp::image::PixelTypeEnum::PTE16uC4 && aDataType != opp::image::PixelTypeEnum::PTE8uC4 &&
        aDataType != opp::image::PixelTypeEnum::PTE32uC4A && aDataType != opp::image::PixelTypeEnum::PTE16uC4A &&
        aDataType != opp::image::PixelTypeEnum::PTE8uC4A)
    {
        throw EXCEPTION("EncodeDifferencingPredictor is only available for Pixel32uC1/2/3/4/4A, Pixel16uC1/2/3/4/4A "
                        "and Pixel8uC1/2/3/4/4A pixeltype.");
    }

    if (aWidth < 2)
    {
        throw INVALIDARGUMENT(aWidth, "The image must be larger than 2 pixels. Given width is: " << aWidth);
    }

    const size_t height = to_size_t(aHeight);
    const size_t width  = to_size_t(aWidth);
    size_t lineOffset   = 0;

    const size_t samplesPerPixel = opp::image::GetChannelCount(aDataType);

    const size_t bitsPerSample = 8 * opp::image::GetPixelSizeInBytes(aDataType) / samplesPerPixel;

    const size_t imageOffset     = 0;
    const size_t nextPixelOffset = samplesPerPixel;
    const size_t colorPlanes     = 1;
    const size_t planeOffset     = 0;
    lineOffset                   = width * samplesPerPixel /* * bitsPerSample / 8*/;

    /*if (mIsPlanar) //can't be planar at this stage
    {
        nextPixelOffset = 1;
        planeOffset     = width * height;
        colorPlanes     = samplesPerPixel;
        lineOffset      = width * bitsPerSample / 8;
    }*/

    switch (bitsPerSample)
    {
        case 8:
        {
            for (size_t line = 0; line < height; line++)
            {
                for (size_t x = width - 1; x >= 1; x--)
                {
                    for (size_t colorPlane = 0; colorPlane < colorPlanes; colorPlane++)
                    {
                        byte *ptr = reinterpret_cast<byte *>(aData);
                        byte *thisPixel =
                            ptr + imageOffset + planeOffset * colorPlane + line * lineOffset + x * nextPixelOffset;
                        byte *previousPixel = thisPixel - nextPixelOffset;
                        for (size_t sample = 0; sample < nextPixelOffset; sample++)
                        {
                            *(thisPixel + sample) -= *(previousPixel + sample);
                        }
                    }
                }
            }
            break;
        }
        case 16: // NOLINT
        {
            for (size_t line = 0; line < height; line++)
            {
                for (size_t x = width - 1; x >= 1; x--)
                {
                    for (size_t colorPlane = 0; colorPlane < colorPlanes; colorPlane++)
                    {
                        ushort *ptr = reinterpret_cast<ushort *>(aData);
                        ushort *thisPixel =
                            ptr + imageOffset + planeOffset * colorPlane + line * lineOffset + x * nextPixelOffset;
                        ushort *previousPixel = thisPixel - nextPixelOffset;
                        for (size_t sample = 0; sample < nextPixelOffset; sample++)
                        {
                            *(thisPixel + sample) -= *(previousPixel + sample);
                        }
                    }
                }
            }
            break;
        }
        case 32: // NOLINT
        {
            for (size_t line = 0; line < height; line++)
            {
                for (size_t x = width - 1; x >= 1; x--)
                {
                    for (size_t colorPlane = 0; colorPlane < colorPlanes; colorPlane++)
                    {
                        uint *ptr = reinterpret_cast<uint *>(aData);
                        uint *thisPixel =
                            ptr + imageOffset + planeOffset * colorPlane + line * lineOffset + x * nextPixelOffset;
                        uint *previousPixel = thisPixel - nextPixelOffset;
                        for (size_t sample = 0; sample < nextPixelOffset; sample++)
                        {
                            *(thisPixel + sample) -= *(previousPixel + sample);
                        }
                    }
                }
            }
            break;
        }
        default:
            throw EXCEPTION("Only TIFF files with unsigned integers and size per sample of 8, 16 "
                            "and 32 bits are supported for differencing prediction.");
            break;
    }
}

void TIFFFile::ReadPlaneNoCompression(size_t aIdx)
{
    for (size_t stripe = 0; stripe < mReadSegments[aIdx].size(); stripe++)
    {
        const size_t destinationOffset = mReadSegments[aIdx][stripe].DestinationOffset;
        size_t toRead                  = mReadSegments[aIdx][stripe].DataSize;
        SeekRead(mReadSegments[aIdx][stripe].DataOffset, std::ios_base::beg);

        size_t imageSizeInBytes = GetImageSizeInBytes();
        if (mIsPlanar)
        {
            imageSizeInBytes *= to_size_t(mSamplesPerPixel);
        }

        // make sure not to run over the end of the image plane
        if (destinationOffset + toRead - aIdx * imageSizeInBytes > imageSizeInBytes)
        {
            toRead = imageSizeInBytes - destinationOffset;
        }
        if (toRead > 0)
        {
            Read(mData.data() + destinationOffset, toRead);
        }

        if (!IsLittleEndian())
        {
            size_t elementSize = to_size_t(mBitsPerSample) / 8;
            size_t dataSize    = toRead / elementSize;

            // complex data types:
            if (mSampleFormat == TIFFSampleFormat::COMPLEXIEEEFP || mSampleFormat == TIFFSampleFormat::COMPLEXINT)
            {
                elementSize /= 2;
                dataSize *= 2;
            }

            EndianSwap(mData.data() + destinationOffset, dataSize, elementSize);
        }
    }
}

void TIFFFile::ReadPlaneLZWCompression(size_t aIdx)
{
    // twice the uncompressed size should be sufficient...
    std::vector<byte> tempBuffer(mReadSegments[aIdx][0].RowsPerStrip * to_size_t(mWidth) * to_size_t(mBitsPerSample) /
                                 8 * to_size_t(mSamplesPerPixel) * 2);

    LZWDecoder lzw;
    size_t colorPlane = 0;
    for (size_t stripe = 0; stripe < mReadSegments[aIdx].size(); stripe++)
    {
        std::fill(tempBuffer.begin(), tempBuffer.end(), byte(0));
        lzw.Reset();

        const size_t offset = mReadSegments[aIdx][stripe].DataOffset;
        const size_t toRead = mReadSegments[aIdx][stripe].DataSize;
        SeekRead(offset, std::ios_base::beg);

        if (toRead > 0)
        {
            Read(reinterpret_cast<char *>(tempBuffer.data()), toRead);
        }

        size_t decodedSize = 0;

        if (mIsPlanar)
        {
            decodedSize = mReadSegments[aIdx][stripe].RowsPerStrip * to_size_t(mWidth) * to_size_t(mBitsPerSample) / 8;
        }
        else
        {
            decodedSize = mReadSegments[aIdx][stripe].RowsPerStrip * to_size_t(mWidth) * to_size_t(mBitsPerSample) / 8 *
                          to_size_t(mSamplesPerPixel);
        }
        const size_t planeSize = GetImageSizeInBytes();
        if (mIsPlanar)
        {
            if (mReadSegments[aIdx][stripe].DestinationOffset + decodedSize -
                    aIdx * planeSize * to_size_t(mSamplesPerPixel) - colorPlane * planeSize >
                planeSize)
            {
                decodedSize = planeSize - mReadSegments[aIdx][stripe].DestinationOffset -
                              (aIdx * planeSize * to_size_t(mSamplesPerPixel) - colorPlane * planeSize);
                colorPlane++;
            }
        }
        else
        {
            if (mReadSegments[aIdx][stripe].DestinationOffset + decodedSize - aIdx * planeSize > planeSize)
            {
                decodedSize = planeSize - mReadSegments[aIdx][stripe].DestinationOffset - aIdx * planeSize;
            }
        }

        if (!lzw.Decode(tempBuffer.data(), decodedSize, mData.data() + mReadSegments[aIdx][stripe].DestinationOffset))
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(FileName(), "Cannot read TIFF file: Error while decoding LZW compression.");
        }

        if (!IsLittleEndian())
        {
            size_t elementSize = to_size_t(mBitsPerSample) / 8;
            size_t dataSize    = decodedSize / elementSize;

            // complex data types:
            if (mSampleFormat == TIFFSampleFormat::COMPLEXIEEEFP || mSampleFormat == TIFFSampleFormat::COMPLEXINT)
            {
                elementSize /= 2;
                dataSize *= 2;
            }

            EndianSwap(mData.data() + mReadSegments[aIdx][stripe].DestinationOffset, dataSize, elementSize);
        }
    }
}

void TIFFFile::ReadPlaneDeflateCompression(size_t aIdx)
{
    // twice the uncompressed size should be sufficient...
    std::vector<byte> tempBuffer(mReadSegments[aIdx][0].RowsPerStrip * to_size_t(mWidth) * to_size_t(mBitsPerSample) /
                                 8 * to_size_t(mSamplesPerPixel) * 2);

    size_t colorPlane = 0;
    for (size_t stripe = 0; stripe < mReadSegments[aIdx].size(); stripe++)
    {
        ZLIBDecoder zlib;
        std::fill(tempBuffer.begin(), tempBuffer.end(), byte(0));

        const size_t offset = mReadSegments[aIdx][stripe].DataOffset;
        const size_t toRead = mReadSegments[aIdx][stripe].DataSize;
        SeekRead(offset, std::ios_base::beg);

        if (toRead > 0)
        {
            Read(reinterpret_cast<char *>(tempBuffer.data()), toRead);
        }

        size_t decodedSize = 0;

        if (mIsPlanar)
        {
            decodedSize = mReadSegments[aIdx][stripe].RowsPerStrip * to_size_t(mWidth) * to_size_t(mBitsPerSample) / 8;
        }
        else
        {
            decodedSize = mReadSegments[aIdx][stripe].RowsPerStrip * to_size_t(mWidth) * to_size_t(mBitsPerSample) / 8 *
                          to_size_t(mSamplesPerPixel);
        }
        const size_t planeSize = GetImageSizeInBytes();
        if (mIsPlanar)
        {
            if (mReadSegments[aIdx][stripe].DestinationOffset + decodedSize -
                    aIdx * planeSize * to_size_t(mSamplesPerPixel) - colorPlane * planeSize >
                planeSize)
            {
                decodedSize = planeSize - mReadSegments[aIdx][stripe].DestinationOffset -
                              (aIdx * planeSize * to_size_t(mSamplesPerPixel) - colorPlane * planeSize);
                colorPlane++;
            }
        }
        else
        {
            if (mReadSegments[aIdx][stripe].DestinationOffset + decodedSize - aIdx * planeSize > planeSize)
            {
                decodedSize = planeSize - mReadSegments[aIdx][stripe].DestinationOffset - aIdx * planeSize;
            }
        }

        try
        {
            zlib.Inflate(tempBuffer.data(), toRead,
                         reinterpret_cast<byte *>(mData.data()) + mReadSegments[aIdx][stripe].DestinationOffset,
                         decodedSize);
        }
        catch (const std::exception &ex)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(
                FileName(),
                "Cannot read TIFF file: Error while decoding DEFLATE compression. Additional info: " << ex.what());
        }

        if (!IsLittleEndian())
        {
            size_t elementSize = to_size_t(mBitsPerSample) / 8;
            size_t dataSize    = decodedSize / elementSize;

            // complex data types:
            if (mSampleFormat == TIFFSampleFormat::COMPLEXIEEEFP || mSampleFormat == TIFFSampleFormat::COMPLEXINT)
            {
                elementSize /= 2;
                dataSize *= 2;
            }

            EndianSwap(mData.data() + mReadSegments[aIdx][stripe].DestinationOffset, dataSize, elementSize);
        }
    }
}

void TIFFFile::ReadPlanePackBitsCompression(size_t aIdx)
{
    // twice the uncompressed size should be sufficient...
    std::vector<byte> tempBuffer(mReadSegments[aIdx][0].RowsPerStrip * to_size_t(mWidth) * to_size_t(mBitsPerSample) /
                                 8 * to_size_t(mSamplesPerPixel) * 2);

    size_t colorPlane = 0;
    for (size_t stripe = 0; stripe < mReadSegments[aIdx].size(); stripe++)
    {
        std::fill(tempBuffer.begin(), tempBuffer.end(), byte(0));

        const size_t offset = mReadSegments[aIdx][stripe].DataOffset;
        const size_t toRead = mReadSegments[aIdx][stripe].DataSize;
        SeekRead(offset, std::ios_base::beg);

        if (toRead > 0)
        {
            Read(reinterpret_cast<char *>(tempBuffer.data()), toRead);
        }

        size_t decodedSize = 0;

        if (mIsPlanar)
        {
            decodedSize = mReadSegments[aIdx][stripe].RowsPerStrip * to_size_t(mWidth) * to_size_t(mBitsPerSample) / 8;
        }
        else
        {
            decodedSize = mReadSegments[aIdx][stripe].RowsPerStrip * to_size_t(mWidth) * to_size_t(mBitsPerSample) / 8 *
                          to_size_t(mSamplesPerPixel);
        }
        const size_t planeSize = GetImageSizeInBytes();
        if (mIsPlanar)
        {
            if (mReadSegments[aIdx][stripe].DestinationOffset + decodedSize -
                    aIdx * planeSize * to_size_t(mSamplesPerPixel) - colorPlane * planeSize >
                planeSize)
            {
                decodedSize = planeSize - mReadSegments[aIdx][stripe].DestinationOffset -
                              (aIdx * planeSize * to_size_t(mSamplesPerPixel) - colorPlane * planeSize);
                colorPlane++;
            }
        }
        else
        {
            if (mReadSegments[aIdx][stripe].DestinationOffset + decodedSize - aIdx * planeSize > planeSize)
            {
                decodedSize = planeSize - mReadSegments[aIdx][stripe].DestinationOffset - aIdx * planeSize;
            }
        }

        sbyte *packedBits  = reinterpret_cast<sbyte *>(tempBuffer.data());
        sbyte *destination = reinterpret_cast<sbyte *>(mData.data()) + mReadSegments[aIdx][stripe].DestinationOffset;

        for (size_t idxEncoded = 0, idxDecoded = 0; idxEncoded < toRead;)
        {
            const sbyte marker = packedBits[idxEncoded];
            if (marker == -128) // should never happen, but if it does, skip to next marker
            {
                idxEncoded++;
            }
            else if (marker < 0) // repeated byte
            {
                const size_t count = to_size_t((-to_int(marker)) + 1);
                for (size_t i = 0; i < count; i++)
                {
                    destination[idxDecoded] = packedBits[idxEncoded + 1];
                    idxDecoded++;
                }
                idxEncoded += 2;
            }
            else if (marker >= 0) // uncompressed data
            {
                const size_t count = to_size_t(marker) + 1;
                for (size_t i = 0; i < count; i++)
                {
                    destination[idxDecoded] = packedBits[idxEncoded + i + 1];
                    idxDecoded++;
                }
                idxEncoded += count + 1;
            }
        }

        if (!IsLittleEndian())
        {
            size_t elementSize = to_size_t(mBitsPerSample) / 8;
            size_t dataSize    = decodedSize / elementSize;

            // complex data types:
            if (mSampleFormat == TIFFSampleFormat::COMPLEXIEEEFP || mSampleFormat == TIFFSampleFormat::COMPLEXINT)
            {
                elementSize /= 2;
                dataSize *= 2;
            }

            EndianSwap(mData.data() + mReadSegments[aIdx][stripe].DestinationOffset, dataSize, elementSize);
        }
    }
}

bool TIFFFile::IsLittleEndian()
{
    return File::IsLittleEndian();
}

bool TIFFFile::CanWriteAs(int aDimX, int aDimY, opp::image::PixelTypeEnum aDatatype)
{
    if (aDimX < 32767 && aDimY < 32767 && aDimX > 0 && aDimY > 0) // NOLINT
    {
        if (aDatatype != opp::image::PixelTypeEnum::Unknown)
        {
            return true;
        }
    }
    return false;
}

bool TIFFFile::CanWriteAs(const Vec2i &aDim, opp::image::PixelTypeEnum aDatatype)
{
    return CanWriteAs(aDim.x, aDim.y, aDatatype);
}

bool TIFFFile::CanWriteAs(const Vec3i &aDim, opp::image::PixelTypeEnum aDatatype)
{
    if (aDim.z != 1)
    {
        return false;
    }
    return CanWriteAs(aDim.x, aDim.y, aDatatype);
}

opp::image::PixelTypeEnum TIFFFile::GetDataTypeUnsigned() const
{
    // NOLINTBEGIN
    if (mSamplesPerPixel == 1 || mIsPlanar)
    {
        if (mBitsPerSample == 8)
        {
            return opp::image::PixelTypeEnum::PTE8uC1;
        }
        if (mBitsPerSample == 16)
        {
            return opp::image::PixelTypeEnum::PTE16uC1;
        }
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE32uC1;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE64uC1;
        }
    }
    if (mSamplesPerPixel == 2)
    {
        if (mBitsPerSample == 8)
        {
            return opp::image::PixelTypeEnum::PTE8uC2;
        }
        if (mBitsPerSample == 16)
        {
            return opp::image::PixelTypeEnum::PTE16uC2;
        }
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE32uC2;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE64uC2;
        }
    }
    if (mSamplesPerPixel == 3)
    {
        if (mBitsPerSample == 8)
        {
            return opp::image::PixelTypeEnum::PTE8uC3;
        }
        if (mBitsPerSample == 16)
        {
            return opp::image::PixelTypeEnum::PTE16uC3;
        }
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE32uC3;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE64uC3;
        }
    }
    if (mSamplesPerPixel == 4)
    {
        if (mBitsPerSample == 8)
        {
            return opp::image::PixelTypeEnum::PTE8uC4;
        }
        if (mBitsPerSample == 16)
        {
            return opp::image::PixelTypeEnum::PTE16uC4;
        }
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE32uC4;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE64uC4;
        }
    }
    return opp::image::PixelTypeEnum::Unknown;
    // NOLINTEND
}

opp::image::PixelTypeEnum TIFFFile::GetDataTypeSigned() const
{
    // NOLINTBEGIN
    if (mSamplesPerPixel == 1 || mIsPlanar)
    {
        if (mBitsPerSample == 8)
        {
            return opp::image::PixelTypeEnum::PTE8sC1;
        }
        if (mBitsPerSample == 16)
        {
            return opp::image::PixelTypeEnum::PTE16sC1;
        }
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE32sC1;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE64sC1;
        }
    }
    if (mSamplesPerPixel == 2)
    {
        if (mBitsPerSample == 8)
        {
            return opp::image::PixelTypeEnum::PTE8sC2;
        }
        if (mBitsPerSample == 16)
        {
            return opp::image::PixelTypeEnum::PTE16sC2;
        }
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE32sC2;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE64sC2;
        }
    }
    if (mSamplesPerPixel == 3)
    {
        if (mBitsPerSample == 8)
        {
            return opp::image::PixelTypeEnum::PTE8sC3;
        }
        if (mBitsPerSample == 16)
        {
            return opp::image::PixelTypeEnum::PTE16sC3;
        }
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE32sC3;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE64sC3;
        }
    }
    if (mSamplesPerPixel == 4)
    {
        if (mBitsPerSample == 8)
        {
            return opp::image::PixelTypeEnum::PTE8sC4;
        }
        if (mBitsPerSample == 16)
        {
            return opp::image::PixelTypeEnum::PTE16sC4;
        }
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE32sC4;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE64sC4;
        }
    }
    return opp::image::PixelTypeEnum::Unknown;
    // NOLINTEND
}

opp::image::PixelTypeEnum TIFFFile::GetDataTypeFloat() const
{
    // NOLINTBEGIN
    if (mSamplesPerPixel == 1 || mIsPlanar)
    {
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE32fC1;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE64fC1;
        }
    }
    if (mSamplesPerPixel == 2)
    {
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE32fC2;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE64fC2;
        }
    }
    if (mSamplesPerPixel == 3)
    {
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE32fC3;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE64fC3;
        }
    }
    if (mSamplesPerPixel == 4)
    {
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE32fC4;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE64fC4;
        }
    }
    return opp::image::PixelTypeEnum::Unknown;
    // NOLINTEND
}

opp::image::PixelTypeEnum TIFFFile::GetDataTypeComplex() const
{
    // NOLINTBEGIN
    if (mSamplesPerPixel == 1 || mIsPlanar)
    {
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE16scC1;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE32scC1;
        }
        if (mBitsPerSample == 128)
        {
            return opp::image::PixelTypeEnum::PTE64scC1;
        }
    }
    if (mSamplesPerPixel == 2)
    {
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE16scC2;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE32scC2;
        }
        if (mBitsPerSample == 128)
        {
            return opp::image::PixelTypeEnum::PTE64scC2;
        }
    }
    if (mSamplesPerPixel == 3)
    {
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE16scC3;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE32scC3;
        }
        if (mBitsPerSample == 128)
        {
            return opp::image::PixelTypeEnum::PTE64scC3;
        }
    }
    if (mSamplesPerPixel == 4)
    {
        if (mBitsPerSample == 32)
        {
            return opp::image::PixelTypeEnum::PTE16scC4;
        }
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE32scC4;
        }
        if (mBitsPerSample == 128)
        {
            return opp::image::PixelTypeEnum::PTE64scC4;
        }
    }
    return opp::image::PixelTypeEnum::Unknown;
    // NOLINTEND
}

opp::image::PixelTypeEnum TIFFFile::GetDataTypeComplexFloat() const
{
    // NOLINTBEGIN
    if (mSamplesPerPixel == 1 || mIsPlanar)
    {
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE32fcC1;
        }
        if (mBitsPerSample == 128)
        {
            return opp::image::PixelTypeEnum::PTE64fcC1;
        }
    }
    if (mSamplesPerPixel == 2)
    {
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE32fcC2;
        }
        if (mBitsPerSample == 128)
        {
            return opp::image::PixelTypeEnum::PTE64fcC2;
        }
    }
    if (mSamplesPerPixel == 3)
    {
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE32fcC3;
        }
        if (mBitsPerSample == 128)
        {
            return opp::image::PixelTypeEnum::PTE64fcC3;
        }
    }
    if (mSamplesPerPixel == 4)
    {
        if (mBitsPerSample == 64)
        {
            return opp::image::PixelTypeEnum::PTE32fcC4;
        }
        if (mBitsPerSample == 128)
        {
            return opp::image::PixelTypeEnum::PTE64fcC4;
        }
    }
    return opp::image::PixelTypeEnum::Unknown;
    // NOLINTEND
}

opp::image::PixelTypeEnum TIFFFile::GetDataType() const
{
    switch (mSampleFormat)
    {
        case TIFFSampleFormat::UINT:
            return GetDataTypeUnsigned();
        case TIFFSampleFormat::INT:
            return GetDataTypeSigned();
        case TIFFSampleFormat::IEEEFP:
            return GetDataTypeFloat();
        case TIFFSampleFormat::VOIDTYPE:
            return GetDataTypeUnsigned(); // in case that the sample format tag is not present we assume unsigned int
        case TIFFSampleFormat::COMPLEXINT:
            return GetDataTypeComplex();
        case TIFFSampleFormat::COMPLEXIEEEFP:
            return GetDataTypeComplexFloat();
        default:
            return opp::image::PixelTypeEnum::Unknown;
    }
}

void TIFFFile::SetDataType(opp::image::PixelTypeEnum aDataType)
{
    // NOLINTBEGIN
    switch (aDataType)
    {
        case opp::image::PixelTypeEnum::PTE64fC1:
            mSampleFormat              = TIFFSampleFormat::IEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64fC2:
            mSampleFormat              = TIFFSampleFormat::IEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64fC3:
            mSampleFormat              = TIFFSampleFormat::IEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 3;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64fC4:
            mSampleFormat              = TIFFSampleFormat::IEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64fC4A:
            mSampleFormat              = TIFFSampleFormat::IEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64fcC1:
            mSampleFormat              = TIFFSampleFormat::COMPLEXIEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 128;
            return;
        case opp::image::PixelTypeEnum::PTE64fcC2:
            mSampleFormat              = TIFFSampleFormat::COMPLEXIEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 128;
            return;
        case opp::image::PixelTypeEnum::PTE64fcC3:
            mSampleFormat              = TIFFSampleFormat::COMPLEXIEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 3;
            mBitsPerSample             = 128;
            return;
        case opp::image::PixelTypeEnum::PTE64fcC4:
            mSampleFormat              = TIFFSampleFormat::COMPLEXIEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 128;
            return;
        case opp::image::PixelTypeEnum::PTE32fC1:
            mSampleFormat              = TIFFSampleFormat::IEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE32fC2:
            mSampleFormat              = TIFFSampleFormat::IEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE32fC3:
            mSampleFormat              = TIFFSampleFormat::IEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 3;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE32fC4:
            mSampleFormat              = TIFFSampleFormat::IEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE32fC4A:
            mSampleFormat              = TIFFSampleFormat::IEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE32fcC1:
            mSampleFormat              = TIFFSampleFormat::COMPLEXIEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE32fcC2:
            mSampleFormat              = TIFFSampleFormat::COMPLEXIEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE32fcC3:
            mSampleFormat              = TIFFSampleFormat::COMPLEXIEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 3;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE32fcC4:
            mSampleFormat              = TIFFSampleFormat::COMPLEXIEEEFP;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64sC1:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64sC2:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64sC3:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 3;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64sC4:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64sC4A:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64scC1:
            mSampleFormat              = TIFFSampleFormat::COMPLEXINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 128;
            return;
        case opp::image::PixelTypeEnum::PTE64scC2:
            mSampleFormat              = TIFFSampleFormat::COMPLEXINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 128;
            return;
        case opp::image::PixelTypeEnum::PTE64scC3:
            mSampleFormat              = TIFFSampleFormat::COMPLEXINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 3;
            mBitsPerSample             = 128;
            return;
        case opp::image::PixelTypeEnum::PTE64scC4:
            mSampleFormat              = TIFFSampleFormat::COMPLEXINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 128;
            return;
        case opp::image::PixelTypeEnum::PTE64uC1:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64uC2:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64uC3:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 3;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64uC4:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE64uC4A:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE32sC1:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE32sC2:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE32sC3:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 3;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE32sC4:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE32sC4A:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE32scC1:
            mSampleFormat              = TIFFSampleFormat::COMPLEXINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE32scC2:
            mSampleFormat              = TIFFSampleFormat::COMPLEXINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE32scC3:
            mSampleFormat              = TIFFSampleFormat::COMPLEXINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 3;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE32scC4:
            mSampleFormat              = TIFFSampleFormat::COMPLEXINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 64;
            return;
        case opp::image::PixelTypeEnum::PTE32uC1:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE32uC2:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE32uC3:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 3;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE32uC4:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE32uC4A:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE16sC1:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 16;
            return;
        case opp::image::PixelTypeEnum::PTE16sC2:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 16;
            return;
        case opp::image::PixelTypeEnum::PTE16sC3:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 3;
            mBitsPerSample             = 16;
            return;
        case opp::image::PixelTypeEnum::PTE16sC4:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 16;
            return;
        case opp::image::PixelTypeEnum::PTE16sC4A:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 16;
            return;
        case opp::image::PixelTypeEnum::PTE16scC1:
            mSampleFormat              = TIFFSampleFormat::COMPLEXINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE16scC2:
            mSampleFormat              = TIFFSampleFormat::COMPLEXINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE16scC3:
            mSampleFormat              = TIFFSampleFormat::COMPLEXINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 3;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE16scC4:
            mSampleFormat              = TIFFSampleFormat::COMPLEXINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 32;
            return;
        case opp::image::PixelTypeEnum::PTE16uC1:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 16;
            return;
        case opp::image::PixelTypeEnum::PTE16uC2:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 16;
            return;
        case opp::image::PixelTypeEnum::PTE16uC3:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 3;
            mBitsPerSample             = 16;
            return;
        case opp::image::PixelTypeEnum::PTE16uC4:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 16;
            return;
        case opp::image::PixelTypeEnum::PTE16uC4A:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 16;
            return;
        case opp::image::PixelTypeEnum::PTE8sC1:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 8;
            return;
        case opp::image::PixelTypeEnum::PTE8sC2:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 8;
            return;
        case opp::image::PixelTypeEnum::PTE8sC3:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 8;
            return;
        case opp::image::PixelTypeEnum::PTE8sC4:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 8;
            return;
        case opp::image::PixelTypeEnum::PTE8sC4A:
            mSampleFormat              = TIFFSampleFormat::INT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 8;
            return;
        case opp::image::PixelTypeEnum::PTE8uC1:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 1;
            mBitsPerSample             = 8;
            return;
        case opp::image::PixelTypeEnum::PTE8uC2:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::BlackIsZero;
            mSamplesPerPixel           = 2;
            mBitsPerSample             = 8;
            return;
        case opp::image::PixelTypeEnum::PTE8uC3:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 3;
            mBitsPerSample             = 8;
            return;
        case opp::image::PixelTypeEnum::PTE8uC4:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 8;
            return;
        case opp::image::PixelTypeEnum::PTE8uC4A:
            mSampleFormat              = TIFFSampleFormat::UINT;
            mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
            mSamplesPerPixel           = 4;
            mBitsPerSample             = 8;
            return;
        default:
            break;
    }

    // NOLINTEND
    throw INVALIDARGUMENT(aDataType, "Cannot convert the provided datatype '" << aDataType << "' to TIFF file format.");
}

size_t TIFFFile::GetImageSizeInBytes() const
{
    if (mIsPlanar)
    {
        return to_size_t(mWidth) * to_size_t(mHeight) * to_size_t(mBitsPerSample) / 8;
    }
    return to_size_t(mWidth) * to_size_t(mHeight) * to_size_t(mSamplesPerPixel) * to_size_t(mBitsPerSample) / 8;
}

size_t TIFFFile::DataSize() const
{
    return to_size_t(mWidth) * to_size_t(mHeight) * to_size_t(mPlanes) * to_size_t(mSamplesPerPixel) *
           to_size_t(mBitsPerSample) / 8;
}

void *TIFFFile::Data()
{
    return mData.data();
}

void *TIFFFile::Data(size_t aIdx)
{
    if (mData.empty())
    {
        return nullptr;
    }

    if (aIdx >= to_size_t(Depth()))
    {
        return nullptr;
    }

    size_t imageSize = GetImageSizeInBytes();
    if (mIsPlanar)
    {
        imageSize *= to_size_t(mSamplesPerPixel);
    }
    return mData.data() + aIdx * imageSize;
}

void *TIFFFile::Data(size_t aIdx, size_t aColorChannel)
{
    if (mData.empty())
    {
        return nullptr;
    }

    if (aIdx >= to_size_t(Depth()))
    {
        return nullptr;
    }

    if (aColorChannel >= to_size_t(mSamplesPerPixel))
    {
        return nullptr;
    }

    if (aColorChannel > 0 && !mIsPlanar)
    {
        return nullptr;
    }

    size_t imageSizeAllColorChannels      = GetImageSizeInBytes();
    const size_t imageSizeOneColorChannel = GetImageSizeInBytes();
    if (mIsPlanar)
    {
        imageSizeAllColorChannels *= to_size_t(mSamplesPerPixel);
    }
    return mData.data() + aIdx * imageSizeAllColorChannels + aColorChannel * imageSizeOneColorChannel;
}

Vec3i TIFFFile::Size() const
{
    return {Width(), Height(), Depth()};
}

opp::image::Size2D TIFFFile::SizePlane() const
{
    return {Width(), Height()};
}

int TIFFFile::Width() const
{
    return mWidth;
}

int TIFFFile::Height() const
{
    return mHeight;
}

int TIFFFile::Depth() const
{
    return mPlanes;
}

int TIFFFile::BitsPerSample() const
{
    return mBitsPerSample;
}

int TIFFFile::SamplesPerPixel() const
{
    return mSamplesPerPixel;
}

bool TIFFFile::IsPlanar() const
{
    return mIsPlanar;
}

tiffTag::TiffOrientation TIFFFile::Orientation() const
{
    return mOrientation;
}

double TIFFFile::PixelSize() const
{
    return mPixelSize;
}

void TIFFFile::SetPixelSize(double aPixelSize)
{
    mPixelSize = aPixelSize;
}

void TIFFFile::ReadSlice(size_t aIdx)
{
    ReadSlices(aIdx, 1);
}

void TIFFFile::ReadSlices(size_t aStartIdx, size_t aSliceCount)
{
    const size_t dataSize = DataSize();

    if (ReadStream() == nullptr)
    {
        throw FILEIOEXCEPTION(FileName(), "The file must be opened with OpenFileForReading() before using ReadSlice()");
    }

    if (!ReadStream()->good())
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(
            FileName(), "Error while reading from file stream. Something must have been wrong before this point.");
    }

    if (dataSize == 0)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(),
                              "The file header is empty. Before calling ReadSlice(), call OpenAndReadHeader().");
    }

    if (aStartIdx + aSliceCount > to_size_t(Depth()) || Depth() < 0)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(
            FileName(),
            "Requested slice indices exceed image size. StartIndex + SliceCount must be <= DimZ of image but got: "
                << aStartIdx << " + " << aSliceCount << " > " << Depth());
    }

    // first call to ReadSlice: allocate the buffer and set it to zero
    if (mData.empty())
    {
        mData.resize(dataSize, 0);
    }
    else
    {
        // make sure that buffer is large enough
        if (mData.size() != dataSize)
        {
            CloseFileForReading();
            throw FILEIOEXCEPTION(
                FileName(), "Something went really wrong: Computed and allocated buffer size do not match. Allocated: "
                                << mData.size() << " expected: " << dataSize);
        }
    }

    for (size_t plane = aStartIdx; plane < aSliceCount; plane++)
    {
        ReadPlane(plane);
    }

    if (!ReadStream()->good())
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(
            FileName(), "Error while reading from file stream. Something must have been wrong before this point.");
    }
}

void TIFFFile::ReadSlice(void *aData, size_t aIdx)
{
    ReadSlices(aData, aIdx, 1);
}

void TIFFFile::ReadSlices(void *aData, size_t aStartIdx, size_t aSliceCount)
{
    const size_t dataSize = DataSize();

    if (dataSize == 0)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(),
                              "The file header is empty. Before calling ReadSlice(), call OpenAndReadHeader().");
    }

    if (aStartIdx + aSliceCount > to_size_t(Depth()) || Depth() < 0)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(
            FileName(),
            "Requested slice indices exceed image size. StartIndex + SliceCount must be <= DimZ of image but got: "
                << aStartIdx << " + " << aSliceCount << " > " << Depth());
    }

    // make sure that internal buffer is filled:
    if (mData.empty())
    {
        ReadSlices(0, to_size_t(Depth()));
    }

    size_t imageSize = GetImageSizeInBytes();
    if (mIsPlanar)
    {
        imageSize *= to_size_t(mSamplesPerPixel);
    }

    const size_t offsetInBuffer = aStartIdx * imageSize;

    const size_t sizeToCopy = aSliceCount * imageSize;

    std::memcpy(aData, mData.data() + offsetInBuffer, sizeToCopy);
}

void TIFFFile::ReadRaw(void *aData, size_t aSizeInBytes, size_t aOffset)
{
    const size_t dataSize = DataSize();

    if (dataSize == 0)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(FileName(),
                              "The file header is empty. Before calling ReadSlice(), call OpenAndReadHeader().");
    }

    if (aSizeInBytes + aOffset > dataSize || Depth() < 0)
    {
        CloseFileForReading();
        throw FILEIOEXCEPTION(
            FileName(), "Requested data range exceeds the file size: SizeInBytes + Offset must be <= dataSize but got: "
                            << aSizeInBytes << " + " << aOffset << " > " << dataSize);
    }

    // make sure that internal buffer is filled:
    if (mData.empty())
    {
        ReadSlices(0, to_size_t(Depth()));
    }

    const size_t offsetInBuffer = aOffset;

    const size_t sizeToCopy = aSizeInBytes;

    std::memcpy(aData, mData.data() + offsetInBuffer, sizeToCopy);
}

FileType TIFFFile::GetFileType() const
{
    return FileType::TIFF;
}

void TIFFFile::WriteTIFF(const std::filesystem::path &aFileName, int aDimX, int aDimY, double aPixelSize,
                         opp::image::PixelTypeEnum aDatatype, const void *aData)
{
    if (!CanWriteAs(aDimX, aDimY, aDatatype))
    {
        throw FILEIOEXCEPTION(aFileName, "Images with dimension " << aDimX << " x " << aDimY << " and data type "
                                                                  << aDatatype << " are not supported for writing.");
    }

    TIFFFile tiff(aFileName);
    tiff.SetDataType(aDatatype);
    tiff.mWidth  = aDimX;
    tiff.mHeight = aDimY;
    tiff.mPlanes = 1;

    Rational pixelSize(720, 10); // NOLINT
    if (aPixelSize != 0)
    {
        pixelSize = TIFFConvertPixelSizeToDPI(aPixelSize);
    }

    tiff.mImageFileDirectories.clear();
    tiff.mImageFileDirectories.push_back(std::make_shared<ImageFileDirectory>(
        to_uint(aDimX), to_uint(aDimY), aPixelSize, to_ushort(tiff.mBitsPerSample), to_ushort(tiff.mSamplesPerPixel),
        tiff.mSampleFormat, false, tiff.mPhotometricInterpretation));

    tiff.OpenFileForWriting(FileOpenMode::EraseOldFile);

    std::array<char, 8> header{0x49, 0x49, 0x2A, 00, 8, 0, 0, 0}; // NOLINT Tiff header with offset to first IFD=8

    tiff.Write(header.data(), header.size());

    const std::shared_ptr<ImageFileDirectory> dir = tiff.mImageFileDirectories[0];

    ushort entryCount = to_ushort(dir->GetEntries().size());

    tiff.Write(reinterpret_cast<char *>(&entryCount), sizeof(ushort));

    for (auto &entry : dir->GetEntries())
    {
        entry->SavePass1(*tiff.GetWriteStream());
    }

    uint newIfd = 0; // marker that no more IFDs are coming
    tiff.Write(reinterpret_cast<char *>(&newIfd), 4);

    for (auto &entry : dir->GetEntries())
    {
        entry->SavePass2(*tiff.GetWriteStream());
    }

    tiff.SeekWrite(0, std::ios_base::end);

    size_t finalImageOffset = tiff.TellWrite();
    finalImageOffset += finalImageOffset % 4;
    std::vector<uint> finallOffsets{to_uint(finalImageOffset)};
    std::dynamic_pointer_cast<IFDStripOffsets>(dir->GetEntry(IFDStripOffsets::TagID))
        ->SaveFinalOffsets(*tiff.GetWriteStream(), finallOffsets);

    tiff.SeekWrite(finalImageOffset, std::ios_base::beg);

    // write finally the image data:
    tiff.Write(reinterpret_cast<const char *>(aData), tiff.DataSize());

    const bool ok = tiff.GetWriteStream()->good();

    tiff.CloseFileForWriting();

    if (!ok)
    {
        throw FILEIOEXCEPTION(aFileName,
                              "Error while writing to file stream. Something must have been wrong before this point.");
    }
}

void TIFFFile::WriteTIFF(const std::filesystem::path &aFileName, int aDimX, int aDimY, double aPixelSize,
                         opp::image::PixelTypeEnum aDatatype, void *aData, int aZIPCompressionLevel)
{
    if (!CanWriteAs(aDimX, aDimY, aDatatype))
    {
        throw FILEIOEXCEPTION(aFileName, "Images with dimension " << aDimX << " x " << aDimY << " and data type "
                                                                  << aDatatype << " are not supported for writing.");
    }
    byte *dataIn = reinterpret_cast<byte *>(aData); // NOLINT

    bool diffEncoded = false;
    // all unsigned int pixel types:
    if (aDatatype == opp::image::PixelTypeEnum::PTE8uC1 || aDatatype == opp::image::PixelTypeEnum::PTE8uC2 ||
        aDatatype == opp::image::PixelTypeEnum::PTE8uC3 || aDatatype == opp::image::PixelTypeEnum::PTE8uC4 ||
        aDatatype == opp::image::PixelTypeEnum::PTE8uC4A || //
        aDatatype == opp::image::PixelTypeEnum::PTE16uC1 || aDatatype == opp::image::PixelTypeEnum::PTE16uC2 ||
        aDatatype == opp::image::PixelTypeEnum::PTE16uC3 || aDatatype == opp::image::PixelTypeEnum::PTE16uC4 ||
        aDatatype == opp::image::PixelTypeEnum::PTE16uC4A || //
        aDatatype == opp::image::PixelTypeEnum::PTE32uC1 || aDatatype == opp::image::PixelTypeEnum::PTE32uC2 ||
        aDatatype == opp::image::PixelTypeEnum::PTE32uC3 || aDatatype == opp::image::PixelTypeEnum::PTE32uC4 ||
        aDatatype == opp::image::PixelTypeEnum::PTE32uC4A)
    {
        EncodeDifferencingPredictor(dataIn, aDatatype, uint(aDimX), uint(aDimY));
        diffEncoded = true;
    }

    size_t compressedSize = 0;
    const size_t dataSize = to_size_t(aDimX * aDimY) * opp::image::GetPixelSizeInBytes(aDatatype);

    ZLIBEncoder zlib(aZIPCompressionLevel);
    std::vector<byte> buffer(dataSize * 2, 0); // * 2 for safety
    compressedSize = zlib.Deflate(dataIn, dataSize, buffer.data(), buffer.size());

    if (compressedSize > buffer.size())
    {
        throw FILEIOEXCEPTION(
            aFileName, "Failed to compress image data, returned buffer size is larger than the provied buffer. Given: "
                           << buffer.size() << " returned value is: " << compressedSize);
    }

    TIFFFile tiff(aFileName);
    tiff.SetDataType(aDatatype);
    tiff.mWidth  = aDimX;
    tiff.mHeight = aDimY;
    tiff.mPlanes = 1;

    Rational pixelSize(720, 10); // NOLINT
    if (aPixelSize != 0)
    {
        pixelSize = TIFFConvertPixelSizeToDPI(aPixelSize);
    }

    tiff.mImageFileDirectories.clear();
    tiff.mImageFileDirectories.push_back(std::make_shared<ImageFileDirectory>(
        to_uint(aDimX), to_uint(aDimY), aPixelSize, to_ushort(tiff.mBitsPerSample), to_ushort(tiff.mSamplesPerPixel),
        tiff.mSampleFormat, false, tiff.mPhotometricInterpretation, diffEncoded,
        std::vector<uint>{uint(compressedSize)}));

    tiff.OpenFileForWriting(FileOpenMode::EraseOldFile);

    std::array<char, 8> header{0x49, 0x49, 0x2A, 00, 8, 0, 0, 0}; // NOLINT Tiff header with offset to first IFD=8

    tiff.Write(header.data(), header.size());

    const std::shared_ptr<ImageFileDirectory> dir = tiff.mImageFileDirectories[0];

    ushort entryCount = to_ushort(dir->GetEntries().size());

    tiff.Write(reinterpret_cast<char *>(&entryCount), sizeof(ushort));

    for (auto &entry : dir->GetEntries())
    {
        entry->SavePass1(*tiff.GetWriteStream());
    }

    uint newIfd = 0; // marker that no more IFDs are coming
    tiff.Write(reinterpret_cast<char *>(&newIfd), 4);

    for (auto &entry : dir->GetEntries())
    {
        entry->SavePass2(*tiff.GetWriteStream());
    }

    tiff.SeekWrite(0, std::ios_base::end);

    size_t finalImageOffset = tiff.TellWrite();
    finalImageOffset += finalImageOffset % 4;
    std::vector<uint> finallOffsets{to_uint(finalImageOffset)};
    std::dynamic_pointer_cast<IFDStripOffsets>(dir->GetEntry(IFDStripOffsets::TagID))
        ->SaveFinalOffsets(*tiff.GetWriteStream(), finallOffsets);

    tiff.SeekWrite(finalImageOffset, std::ios_base::beg);

    // write finally the image data:
    tiff.Write(reinterpret_cast<const char *>(buffer.data()), compressedSize);

    const bool ok = tiff.GetWriteStream()->good();

    tiff.CloseFileForWriting();

    if (!ok)
    {
        throw FILEIOEXCEPTION(aFileName,
                              "Error while writing to file stream. Something must have been wrong before this point.");
    }
}

void opp::fileIO::TIFFFile::WriteTIFF(const std::filesystem::path &aFileName, int aDimX, int aDimY, double aPixelSize,
                                      opp::image::PixelTypeEnum aDatatype, const void *aData0, const void *aData1,
                                      const void *aData2, const void *aData3)
{
    if (GetChannelCount(aDatatype) > 1)
    {
        throw INVALIDARGUMENT(
            aDatatype,
            "Only single channel images can be merged to a planar TIFF file. But provided pixel type is " << aDatatype);
    }

    if (aData0 == nullptr)
    {
        throw INVALIDARGUMENT(aData0, "First data plane cannot be nullptr.");
    }
    int countColorChannels = 1;

    if (aData1 != nullptr)
    {
        countColorChannels = 2;
    }

    if (aData2 != nullptr)
    {
        if (aData1 == nullptr)
        {
            throw INVALIDARGUMENT(aData1, "Provided data pointers must be consecutive. Provided aData2 but "
                                          "aData1 is nullptr.");
        }
        countColorChannels = 3;
    }

    if (aData3 != nullptr)
    {
        if (aData1 == nullptr || aData2 == nullptr)
        {
            throw INVALIDARGUMENT(aData1 or aData2, "Provided data pointers must be consecutive. Provided aData3 but "
                                                    "at least one of aData1 and aData2 is nullptr.");
        }
        countColorChannels = 4;
    }

    if (!CanWriteAs(aDimX, aDimY, aDatatype))
    {
        throw FILEIOEXCEPTION(aFileName, "Images with dimension " << aDimX << " x " << aDimY << " and data type "
                                                                  << aDatatype << " are not supported for writing.");
    }

    TIFFFile tiff(aFileName);
    tiff.SetDataType(aDatatype);
    tiff.mSamplesPerPixel = countColorChannels;
    tiff.mWidth           = aDimX;
    tiff.mHeight          = aDimY;
    tiff.mPlanes          = 1;
    tiff.mIsPlanar        = true;
    if (countColorChannels > 2)
    {
        tiff.mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
    }

    Rational pixelSize(720, 10); // NOLINT
    if (aPixelSize != 0)
    {
        pixelSize = TIFFConvertPixelSizeToDPI(aPixelSize);
    }

    tiff.mImageFileDirectories.clear();
    tiff.mImageFileDirectories.push_back(std::make_shared<ImageFileDirectory>(
        to_uint(aDimX), to_uint(aDimY), aPixelSize, to_ushort(tiff.mBitsPerSample), to_ushort(tiff.mSamplesPerPixel),
        tiff.mSampleFormat, true, tiff.mPhotometricInterpretation));

    tiff.OpenFileForWriting(FileOpenMode::EraseOldFile);

    std::array<char, 8> header{0x49, 0x49, 0x2A, 00, 8, 0, 0, 0}; // NOLINT Tiff header with offset to first IFD=8

    tiff.Write(header.data(), header.size());

    const std::shared_ptr<ImageFileDirectory> dir = tiff.mImageFileDirectories[0];

    ushort entryCount = to_ushort(dir->GetEntries().size());

    tiff.Write(reinterpret_cast<char *>(&entryCount), sizeof(ushort));

    for (auto &entry : dir->GetEntries())
    {
        entry->SavePass1(*tiff.GetWriteStream());
    }

    uint newIfd = 0; // marker that no more IFDs are coming
    tiff.Write(reinterpret_cast<char *>(&newIfd), 4);

    for (auto &entry : dir->GetEntries())
    {
        entry->SavePass2(*tiff.GetWriteStream());
    }

    tiff.SeekWrite(0, std::ios_base::end);

    size_t finalImageOffset0 = tiff.TellWrite();
    finalImageOffset0 += finalImageOffset0 % 4;
    std::vector<uint> finallOffsets{to_uint(finalImageOffset0)};

    for (int colorChannel = 1; colorChannel < countColorChannels; colorChannel++)
    {
        const size_t offset = finalImageOffset0 + to_size_t(colorChannel) * tiff.GetImageSizeInBytes();
        finallOffsets.push_back(to_uint(offset));
    }

    std::dynamic_pointer_cast<IFDStripOffsets>(dir->GetEntry(IFDStripOffsets::TagID))
        ->SaveFinalOffsets(*tiff.GetWriteStream(), finallOffsets);

    tiff.SeekWrite(finalImageOffset0, std::ios_base::beg);

    // write finally the image data:
    tiff.Write(reinterpret_cast<const char *>(aData0), tiff.GetImageSizeInBytes());
    if (aData1 != nullptr)
    {
        tiff.Write(reinterpret_cast<const char *>(aData1), tiff.GetImageSizeInBytes());
    }
    if (aData2 != nullptr)
    {
        tiff.Write(reinterpret_cast<const char *>(aData2), tiff.GetImageSizeInBytes());
    }
    if (aData3 != nullptr)
    {
        tiff.Write(reinterpret_cast<const char *>(aData3), tiff.GetImageSizeInBytes());
    }

    const bool ok = tiff.GetWriteStream()->good();

    tiff.CloseFileForWriting();

    if (!ok)
    {
        throw FILEIOEXCEPTION(aFileName,
                              "Error while writing to file stream. Something must have been wrong before this point.");
    }
}

void fileIO::TIFFFile::WriteTIFF(const std::filesystem::path &aFileName, int aDimX, int aDimY, double aPixelSize,
                                 opp::image::PixelTypeEnum aDatatype, void *aData0, void *aData1, void *aData2,
                                 void *aData3, int aZIPCompressionLevel)
{
    if (GetChannelCount(aDatatype) > 1)
    {
        throw INVALIDARGUMENT(
            aDatatype,
            "Only single channel images can be merged to a planar TIFF file. But provided pixel type is " << aDatatype);
    }

    if (aData0 == nullptr)
    {
        throw INVALIDARGUMENT(aData0, "First data plane cannot be nullptr.");
    }
    int countColorChannels = 1;

    if (aData1 != nullptr)
    {
        countColorChannels = 2;
    }

    if (aData2 != nullptr)
    {
        if (aData1 == nullptr)
        {
            throw INVALIDARGUMENT(aData1, "Provided data pointers must be consecutive. Provided aData2 but "
                                          "aData1 is nullptr.");
        }
        countColorChannels = 3;
    }

    if (aData3 != nullptr)
    {
        if (aData1 == nullptr || aData2 == nullptr)
        {
            throw INVALIDARGUMENT(aData1 or aData2, "Provided data pointers must be consecutive. Provided aData3 but "
                                                    "at least one of aData1 and aData2 is nullptr.");
        }
        countColorChannels = 4;
    }

    if (!CanWriteAs(aDimX, aDimY, aDatatype))
    {
        throw FILEIOEXCEPTION(aFileName, "Images with dimension " << aDimX << " x " << aDimY << " and data type "
                                                                  << aDatatype << " are not supported for writing.");
    }
    byte *dataIn0 = reinterpret_cast<byte *>(aData0); // NOLINT
    byte *dataIn1 = reinterpret_cast<byte *>(aData1); // NOLINT
    byte *dataIn2 = reinterpret_cast<byte *>(aData2); // NOLINT
    byte *dataIn3 = reinterpret_cast<byte *>(aData3); // NOLINT

    bool diffEncoded = false;
    // all unsigned int pixel types:
    if (aDatatype == opp::image::PixelTypeEnum::PTE8uC1 ||  //
        aDatatype == opp::image::PixelTypeEnum::PTE16uC1 || //
        aDatatype == opp::image::PixelTypeEnum::PTE32uC1)
    {
        EncodeDifferencingPredictor(dataIn0, aDatatype, uint(aDimX), uint(aDimY));
        if (dataIn1 != nullptr)
        {
            EncodeDifferencingPredictor(dataIn1, aDatatype, uint(aDimX), uint(aDimY));
        }
        if (dataIn2 != nullptr)
        {
            EncodeDifferencingPredictor(dataIn2, aDatatype, uint(aDimX), uint(aDimY));
        }
        if (dataIn3 != nullptr)
        {
            EncodeDifferencingPredictor(dataIn3, aDatatype, uint(aDimX), uint(aDimY));
        }
        diffEncoded = true;
    }

    size_t compressedSize0 = 0;
    size_t compressedSize1 = 0;
    size_t compressedSize2 = 0;
    size_t compressedSize3 = 0;
    std::vector<uint> compressedSizes(to_size_t(countColorChannels));
    const size_t dataSize = to_size_t(aDimX * aDimY) * opp::image::GetPixelSizeInBytes(aDatatype);

    ZLIBEncoder zlib(aZIPCompressionLevel);
    std::vector<byte> buffer0(dataSize * 2, 0); // * 2 for safety
    std::vector<byte> buffer1(dataSize * 2, 0); // * 2 for safety
    std::vector<byte> buffer2(dataSize * 2, 0); // * 2 for safety
    std::vector<byte> buffer3(dataSize * 2, 0); // * 2 for safety
    compressedSize0    = zlib.Deflate(dataIn0, dataSize, buffer0.data(), buffer0.size());
    compressedSizes[0] = to_uint(compressedSize0);
    if (dataIn1 != nullptr)
    {
        ZLIBEncoder zlib2(aZIPCompressionLevel);
        compressedSize1    = zlib2.Deflate(dataIn1, dataSize, buffer1.data(), buffer1.size());
        compressedSizes[1] = to_uint(compressedSize1);
    }
    if (dataIn2 != nullptr)
    {
        ZLIBEncoder zlib2(aZIPCompressionLevel);
        compressedSize2    = zlib2.Deflate(dataIn2, dataSize, buffer2.data(), buffer2.size());
        compressedSizes[2] = to_uint(compressedSize2);
    }
    if (dataIn3 != nullptr)
    {
        ZLIBEncoder zlib2(aZIPCompressionLevel);
        compressedSize3    = zlib2.Deflate(dataIn3, dataSize, buffer3.data(), buffer3.size());
        compressedSizes[3] = to_uint(compressedSize3);
    }

    if (compressedSize0 > buffer0.size())
    {
        throw FILEIOEXCEPTION(
            aFileName, "Failed to compress image data, returned buffer size is larger than the provied buffer. Given: "
                           << buffer0.size() << " returned value is: " << compressedSize0);
    }

    if (dataIn1 != nullptr && compressedSize1 > buffer1.size())
    {
        throw FILEIOEXCEPTION(
            aFileName, "Failed to compress image data, returned buffer size is larger than the provied buffer. Given: "
                           << buffer1.size() << " returned value is: " << compressedSize1);
    }

    if (dataIn2 != nullptr && compressedSize2 > buffer2.size())
    {
        throw FILEIOEXCEPTION(
            aFileName, "Failed to compress image data, returned buffer size is larger than the provied buffer. Given: "
                           << buffer2.size() << " returned value is: " << compressedSize2);
    }

    if (dataIn3 != nullptr && compressedSize3 > buffer3.size())
    {
        throw FILEIOEXCEPTION(
            aFileName, "Failed to compress image data, returned buffer size is larger than the provied buffer. Given: "
                           << buffer3.size() << " returned value is: " << compressedSize3);
    }

    TIFFFile tiff(aFileName);
    tiff.SetDataType(aDatatype);
    tiff.mSamplesPerPixel = countColorChannels;
    tiff.mWidth           = aDimX;
    tiff.mHeight          = aDimY;
    tiff.mPlanes          = 1;
    tiff.mIsPlanar        = true;
    if (countColorChannels > 2)
    {
        tiff.mPhotometricInterpretation = TIFFPhotometricInterpretation::RGB;
    }

    Rational pixelSize(720, 10); // NOLINT
    if (aPixelSize != 0)
    {
        pixelSize = TIFFConvertPixelSizeToDPI(aPixelSize);
    }

    tiff.mImageFileDirectories.clear();
    tiff.mImageFileDirectories.push_back(std::make_shared<ImageFileDirectory>(
        to_uint(aDimX), to_uint(aDimY), aPixelSize, to_ushort(tiff.mBitsPerSample), to_ushort(tiff.mSamplesPerPixel),
        tiff.mSampleFormat, true, tiff.mPhotometricInterpretation, diffEncoded, compressedSizes));

    tiff.OpenFileForWriting(FileOpenMode::EraseOldFile);

    std::array<char, 8> header{0x49, 0x49, 0x2A, 00, 8, 0, 0, 0}; // NOLINT Tiff header with offset to first IFD=8

    tiff.Write(header.data(), header.size());

    const std::shared_ptr<ImageFileDirectory> dir = tiff.mImageFileDirectories[0];

    ushort entryCount = to_ushort(dir->GetEntries().size());

    tiff.Write(reinterpret_cast<char *>(&entryCount), sizeof(ushort));

    for (auto &entry : dir->GetEntries())
    {
        entry->SavePass1(*tiff.GetWriteStream());
    }

    uint newIfd = 0; // marker that no more IFDs are coming
    tiff.Write(reinterpret_cast<char *>(&newIfd), 4);

    for (auto &entry : dir->GetEntries())
    {
        entry->SavePass2(*tiff.GetWriteStream());
    }

    tiff.SeekWrite(0, std::ios_base::end);

    size_t finalImageOffset0 = tiff.TellWrite();
    finalImageOffset0 += finalImageOffset0 % 4;
    std::vector<uint> finallOffsets{to_uint(finalImageOffset0)};

    for (int colorChannel = 1; colorChannel < countColorChannels; colorChannel++)
    {
        size_t offset = finalImageOffset0;
        for (int i = 0; i < colorChannel; i++)
        {
            offset += compressedSizes[to_size_t(i)];
        }
        finallOffsets.push_back(to_uint(offset));
    }

    std::dynamic_pointer_cast<IFDStripOffsets>(dir->GetEntry(IFDStripOffsets::TagID))
        ->SaveFinalOffsets(*tiff.GetWriteStream(), finallOffsets);

    tiff.SeekWrite(finalImageOffset0, std::ios_base::beg);

    // write finally the image data:
    tiff.Write(reinterpret_cast<const char *>(buffer0.data()), compressedSize0);
    if (aData1 != nullptr)
    {
        tiff.Write(reinterpret_cast<const char *>(buffer1.data()), compressedSize1);
    }
    if (aData2 != nullptr)
    {
        tiff.Write(reinterpret_cast<const char *>(buffer2.data()), compressedSize2);
    }
    if (aData3 != nullptr)
    {
        tiff.Write(reinterpret_cast<const char *>(buffer3.data()), compressedSize3);
    }

    const bool ok = tiff.GetWriteStream()->good();

    tiff.CloseFileForWriting();

    if (!ok)
    {
        throw FILEIOEXCEPTION(aFileName,
                              "Error while writing to file stream. Something must have been wrong before this point.");
    }
}
} // namespace opp::fileIO