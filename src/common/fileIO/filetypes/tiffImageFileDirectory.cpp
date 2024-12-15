// --> tiffFile.h must be included first
// clang-format off
#include "tiffFile.h"
#include "tiffImageFileDirectory.h"
#include "tiffImageFileDirectory_impl.h" //NOLINT
// clang-format on
#include <algorithm>
#include <array>
#include <common/defines.h>
#include <common/fileIO/file.h>
#include <common/fileIO/fileWriter.h>
#include <common/fileIO/filetypes/tiffImageFileDirectory.h>
#include <common/fileIO/pseudoFileReader.h>
#include <common/safeCast.h>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <ios>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace opp::fileIO::tiffTag
{
Rational TIFFConvertPixelSizeToDPI(double aPixelSize)
{
    // convert nm to cm to inch
    const double dpi = 1.0 / (aPixelSize / 2.54 / 10E6); // NOLINT
    return {to_uint(dpi), 1};
}

double TIFFConvertDPIToPixelSize(const Rational &aDPI)
{
    // SerialEM stores pixel size as DPI -> convert inch to cm to nm
    return 1.0 / aDPI.GetValue() * 2.54 * 10E6; // NOLINT
}

size_t GetTiffTypeSizeInBytes(TiffType aType)
{
    switch (aType)
    {
        case TiffType::BYTE:
        case TiffType::ASCII:
        case TiffType::SBYTE:
        case TiffType::UNDEFINED:
            return 1;
        case TiffType::SHORT:
        case TiffType::SSHORT:
            return 2;
        case TiffType::LONG:
        case TiffType::SLONG:
        case TiffType::FLOAT:
            return 4;
        case TiffType::RATIONAL:
        case TiffType::SRATIONAL:
        case TiffType::DOUBLE:
        case TiffType::LONG8:
        case TiffType::SLONG8:
        case TiffType::IFD8:
            return 8;
        default:
            return 1;
    };
}

Rational::Rational(uint aNominator, uint aDenominator) : nominator(aNominator), denominator(aDenominator) // NOLINT
{
}

Rational::Rational(uint aValues[2]) : nominator(aValues[0]), denominator(aValues[1])
{
}

double Rational::GetValue() const
{
    return nominator / to_double(denominator);
}

SRational::SRational(int aNominator, int aDenominator) : nominator(aNominator), denominator(aDenominator) // NOLINT
{
}

SRational::SRational(int aValues[2]) : nominator(aValues[0]), denominator(aValues[1])
{
}

double SRational::GetValue() const
{
    return nominator / to_double(denominator);
}

ImageFileDirectory::ImageFileDirectory(TIFFFile &aFile)
{
    mEntryCount = aFile.ReadLE<ushort>(); // NOLINT(cppcoreguidelines-prefer-member-initializer)
    for (ushort i = 0; i < mEntryCount; i++)
    {
        mEntries.push_back(ImageFileDirectoryEntry::CreateFileDirectoryEntry(aFile));
    }
}

ImageFileDirectory::ImageFileDirectory(uint aWidth, uint aHeight, double aPixelSize, ushort aBitPerSample, // NOLINT
                                       ushort aSamplesPerPixel, TIFFSampleFormat aSampleFormat, bool aPlanar,
                                       TIFFPhotometricInterpretation aPhotometricInterpretation)
{
    Rational pixelSize(720, 10); // NOLINT
    if (aPixelSize != 0)
    {
        pixelSize = TIFFConvertPixelSizeToDPI(aPixelSize);
    }

    mEntries.push_back(std::make_shared<IFDImageWidth>(aWidth));
    mEntries.push_back(std::make_shared<IFDImageLength>(aHeight));
    mEntries.push_back(std::make_shared<IFDBitsPerSample>(aBitPerSample));
    mEntries.push_back(std::make_shared<IFDSampleFormat>(aSampleFormat));
    mEntries.push_back(std::make_shared<IFDPlanarConfiguration>(aPlanar ? TIFFPlanarConfigurartion::Planar
                                                                        : TIFFPlanarConfigurartion::Chunky));
    mEntries.push_back(std::make_shared<IFDCompression>(TIFFCompression::NoCompression));
    mEntries.push_back(std::make_shared<IFDPhotometricInterpretation>(aPhotometricInterpretation));
    mEntries.push_back(std::make_shared<IFDStripOffsets>(aPlanar ? to_size_t(aSamplesPerPixel) : 1UL));
    mEntries.push_back(std::make_shared<IFDSamplesPerPixel>(aSamplesPerPixel));
    mEntries.push_back(std::make_shared<IFDRowsPerStrip>(aHeight));
    std::vector<uint> byteCounts;
    if (aPlanar)
    {
        for (ushort i = 0; i < aSamplesPerPixel; i++)
        {
            byteCounts.push_back(aWidth * aHeight * aBitPerSample / 8);
        }
    }
    else
    {
        byteCounts.push_back(aWidth * aHeight * aSamplesPerPixel * aBitPerSample / 8);
    }
    mEntries.push_back(std::make_shared<IFDStripByteCounts>(std::move(byteCounts)));
    mEntries.push_back(std::make_shared<IFDXResolution>(pixelSize));
    mEntries.push_back(std::make_shared<IFDYResolution>(pixelSize));
    mEntries.push_back(std::make_shared<IFDResolutionUnit>(TIFFResolutionUnit::Inch));
    mEntries.push_back(std::make_shared<IFDSoftware>(FILE_CREATED_BY));
    mEntryCount = to_ushort(mEntries.size());
}

ImageFileDirectory::ImageFileDirectory(uint aWidth, uint aHeight, double aPixelSize, ushort aBitPerSample, // NOLINT
                                       ushort aSamplesPerPixel, TIFFSampleFormat aSampleFormat, bool aPlanar,
                                       TIFFPhotometricInterpretation aPhotometricInterpretation, bool aDifference,
                                       std::vector<uint> aCompressedSize)
{
    Rational pixelSize(720, 10); // NOLINT
    if (aPixelSize != 0)
    {
        pixelSize = TIFFConvertPixelSizeToDPI(aPixelSize);
    }

    mEntries.push_back(std::make_shared<IFDImageWidth>(aWidth));
    mEntries.push_back(std::make_shared<IFDImageLength>(aHeight));
    mEntries.push_back(std::make_shared<IFDBitsPerSample>(aBitPerSample));
    mEntries.push_back(std::make_shared<IFDSampleFormat>(aSampleFormat));
    mEntries.push_back(std::make_shared<IFDPlanarConfiguration>(aPlanar ? TIFFPlanarConfigurartion::Planar
                                                                        : TIFFPlanarConfigurartion::Chunky));
    mEntries.push_back(std::make_shared<IFDCompression>(TIFFCompression::Deflate));
    mEntries.push_back(std::make_shared<IFDDifferencingPredictor>(
        aDifference ? TIFFDifferencingPredictor::HorizontalDifferencing : TIFFDifferencingPredictor::None));
    mEntries.push_back(std::make_shared<IFDPhotometricInterpretation>(aPhotometricInterpretation));
    mEntries.push_back(std::make_shared<IFDStripOffsets>(aPlanar ? to_size_t(aSamplesPerPixel) : 1UL));
    mEntries.push_back(std::make_shared<IFDSamplesPerPixel>(aSamplesPerPixel));
    mEntries.push_back(std::make_shared<IFDRowsPerStrip>(aHeight));
    mEntries.push_back(std::make_shared<IFDStripByteCounts>(std::move(aCompressedSize)));
    mEntries.push_back(std::make_shared<IFDXResolution>(pixelSize));
    mEntries.push_back(std::make_shared<IFDYResolution>(pixelSize));
    mEntries.push_back(std::make_shared<IFDResolutionUnit>(TIFFResolutionUnit::Inch));
    mEntries.push_back(std::make_shared<IFDSoftware>(FILE_CREATED_BY));
    mEntryCount = to_ushort(mEntries.size());
}

void ImageFileDirectory::SaveAsTiff(std::ofstream &aStream, void *aData, size_t aDataSize)
{
    std::array<char, 8> header{0x49, 0x49, 0x2A, 00, 8, 0, 0, 0}; // NOLINT Tiff header with offset to first IFD=8

    aStream.write(header.data(), std::streamsize(header.size()));

    ushort entryCount = to_ushort(mEntries.size());

    aStream.write(reinterpret_cast<char *>(&entryCount), 2);

    for (auto &entry : mEntries)
    {
        aStream.flush();
        entry->SavePass1(aStream);
    }

    uint newIfd = 0; // marker that no more IFDs are coming
    aStream.write(reinterpret_cast<char *>(&newIfd), 4);

    for (auto &entry : mEntries)
    {
        aStream.flush();
        entry->SavePass2(aStream);
    }

    aStream.flush();
    aStream.seekp(0, std::ios_base::end);

    std::streampos finalImageOffset = aStream.tellp();
    finalImageOffset += finalImageOffset % 4;
    std::vector<uint> finallOffsets{to_uint(size_t(finalImageOffset))};
    std::dynamic_pointer_cast<IFDStripOffsets>(GetEntry(IFDStripOffsets::TagID))
        ->SaveFinalOffsets(aStream, finallOffsets);

    aStream.seekp(finalImageOffset, std::ios_base::beg);

    aStream.write(reinterpret_cast<char *>(aData), std::streamsize(aDataSize));
}

std::shared_ptr<ImageFileDirectoryEntry> ImageFileDirectory::GetEntry(ushort aTagID)
{
    for (auto &entry : mEntries)
    {
        if (entry->mTag.TagID == aTagID)
        {
            return entry;
        }
    }
    return nullptr;
}

ImageFileDirectoryEntry::ImageFileDirectoryEntry(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), PseudoFileReader(aFile.ReadStream()), mTag(), mOffsetInStream(0)
{
    mTag.TagID          = aTagID;
    mTag.Type           = ReadLE<TiffType>();
    mTag.Count          = ReadLE<uint>();
    mTag.Offset.UIntVal = ReadLE<uint>(); // NOLINT
}

ImageFileDirectoryEntry::ImageFileDirectoryEntry(ushort aTagID, TiffType aFieldType, uint aValueCount)
    : File("", true), mTag(), mOffsetInStream(0)
{
    mTag.TagID          = aTagID;
    mTag.Type           = aFieldType;
    mTag.Count          = aValueCount;
    mTag.Offset.UIntVal = 0; // NOLINT
}

size_t ImageFileDirectoryEntry::WriteEntryHeader(uint aOffsetOrValue, std::ostream &aStream, int aValueCount)
{
    aStream.write(reinterpret_cast<char *>(&mTag.TagID), 2);
    aStream.write(reinterpret_cast<char *>(&mTag.Type), 2);
    if (aValueCount == -1)
    {
        aStream.write(reinterpret_cast<char *>(&mTag.Count), 4);
    }
    else
    {
        aStream.write(reinterpret_cast<char *>(&aValueCount), 4);
    }

    const size_t streamPosition = to_size_t(aStream.tellp());
    aStream.write(reinterpret_cast<char *>(&aOffsetOrValue), sizeof(aOffsetOrValue));

    return streamPosition;
}

void ImageFileDirectoryEntry::WritePass2(const char *aData, size_t aDataLength, std::ostream &aStream)
{
    aStream.seekp(0, std::ios_base::end);
    uint offset = to_uint(size_t(aStream.tellp()));
    offset += offset % 4; // align offsets to largest word boundary
    aStream.seekp(offset, std::ios_base::beg);

    aStream.write(aData, std::streamsize(aDataLength));

    aStream.seekp(std::streamoff(mOffsetInStream), std::ios_base::beg);
    aStream.write(reinterpret_cast<char *>(&offset), sizeof(offset));

    aStream.seekp(0, std::ios_base::end);
}

ImageFileDirectoryEntry::ImageFileDirectoryEntry(TIFFFile &aFile)
    : File("", aFile.IsLittleEndian()), PseudoFileReader(aFile.ReadStream()), mTag(), mOffsetInStream(0)
{
    mTag.TagID          = ReadLE<ushort>();
    mTag.Type           = ReadLE<TiffType>();
    mTag.Count          = ReadLE<uint>();
    mTag.Offset.UIntVal = ReadLE<uint>(); // NOLINT
}

std::shared_ptr<ImageFileDirectoryEntry> ImageFileDirectoryEntry::CreateFileDirectoryEntry(TIFFFile &aFile)
{
    const ushort tagID = aFile.ReadLE<ushort>();
    switch (tagID)
    {
        case IFDArtist::TagID:
            return std::make_shared<IFDArtist>(aFile, tagID);
        case IFDBitsPerSample::TagID:
            return std::make_shared<IFDBitsPerSample>(aFile, tagID);
        case IFDCellLength::TagID:
            return std::make_shared<IFDCellLength>(aFile, tagID);
        case IFDCellWidth::TagID:
            return std::make_shared<IFDCellWidth>(aFile, tagID);
        case IFDColorMap::TagID:
            return std::make_shared<IFDColorMap>(aFile, tagID);
        case IFDCompression::TagID:
            return std::make_shared<IFDCompression>(aFile, tagID);
        case IFDCopyright::TagID:
            return std::make_shared<IFDCopyright>(aFile, tagID);
        case IFDDateTime::TagID:
            return std::make_shared<IFDDateTime>(aFile, tagID);
        case IFDExtraSamples::TagID:
            return std::make_shared<IFDExtraSamples>(aFile, tagID);
        case IFDDifferencingPredictor::TagID:
            return std::make_shared<IFDDifferencingPredictor>(aFile, tagID);
        case IFDFillOrder::TagID:
            return std::make_shared<IFDFillOrder>(aFile, tagID);
        case IFDFreeByteCounts::TagID:
            return std::make_shared<IFDFreeByteCounts>(aFile, tagID);
        case IFDFreeOffsets::TagID:
            return std::make_shared<IFDFreeOffsets>(aFile, tagID);
        case IFDGrayResponseCurve::TagID:
            return std::make_shared<IFDGrayResponseCurve>(aFile, tagID);
        case IFDGrayResponseUnit::TagID:
            return std::make_shared<IFDGrayResponseUnit>(aFile, tagID);
        case IFDHostComputer::TagID:
            return std::make_shared<IFDHostComputer>(aFile, tagID);
        case IFDImageDescription::TagID:
            return std::make_shared<IFDImageDescription>(aFile, tagID);
        case IFDImageLength::TagID:
            return std::make_shared<IFDImageLength>(aFile, tagID);
        case IFDImageWidth::TagID:
            return std::make_shared<IFDImageWidth>(aFile, tagID);
        case IFDMake::TagID:
            return std::make_shared<IFDMake>(aFile, tagID);
        case IFDMaxSampleValue::TagID:
            return std::make_shared<IFDMaxSampleValue>(aFile, tagID);
        case IFDMinSampleValue::TagID:
            return std::make_shared<IFDMinSampleValue>(aFile, tagID);
        case IFDModel::TagID:
            return std::make_shared<IFDModel>(aFile, tagID);
        case IFDNewSubfileType::TagID:
            return std::make_shared<IFDNewSubfileType>(aFile, tagID);
        case IFDOrientation::TagID:
            return std::make_shared<IFDOrientation>(aFile, tagID);
        case IFDPhotometricInterpretation::TagID:
            return std::make_shared<IFDPhotometricInterpretation>(aFile, tagID);
        case IFDPlanarConfiguration::TagID:
            return std::make_shared<IFDPlanarConfiguration>(aFile, tagID);
        case IFDResolutionUnit::TagID:
            return std::make_shared<IFDResolutionUnit>(aFile, tagID);
        case IFDRowsPerStrip::TagID:
            return std::make_shared<IFDRowsPerStrip>(aFile, tagID);
        case IFDSamplesPerPixel::TagID:
            return std::make_shared<IFDSamplesPerPixel>(aFile, tagID);
        case IFDSampleFormat::TagID:
            return std::make_shared<IFDSampleFormat>(aFile, tagID);
        case IFDSoftware::TagID:
            return std::make_shared<IFDSoftware>(aFile, tagID);
        case IFDStripByteCounts::TagID:
            return std::make_shared<IFDStripByteCounts>(aFile, tagID);
        case IFDStripOffsets::TagID:
            return std::make_shared<IFDStripOffsets>(aFile, tagID);
        case IFDSubfileType::TagID:
            return std::make_shared<IFDSubfileType>(aFile, tagID);
        case IFDThreshholding::TagID:
            return std::make_shared<IFDThreshholding>(aFile, tagID);
        case IFDXResolution::TagID:
            return std::make_shared<IFDXResolution>(aFile, tagID);
        case IFDYResolution::TagID:
            return std::make_shared<IFDYResolution>(aFile, tagID);
        default:
            return std::shared_ptr<ImageFileDirectoryEntry>(new ImageFileDirectoryEntry(aFile, tagID));
    }
}

void ImageFileDirectoryEntry::SavePass1(std::ostream & /*aStream*/)
{
    // do nothing
}

void ImageFileDirectoryEntry::SavePass2(std::ostream & /*aStream*/)
{
    // do nothing
}

IFDImageLength::IFDImageLength(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), ImageFileDirectoryEntry(aFile, aTagID)
{
    if (GetTiffTypeSizeInBytes(mTag.Type) == 2 && !IsLittleEndian())
    {
        uint *ptr     = &mTag.Offset.UIntVal; // NOLINT
        ushort *ptrUS = reinterpret_cast<ushort *>(ptr);
        mValue        = ptrUS[4 / GetTiffTypeSizeInBytes(mTag.Type) - 1];
    }
    else
    {
        mValue = mTag.Offset.UIntVal; // NOLINT
    }
}

IFDImageLength::IFDImageLength(uint aValue)
    : File("", true), ImageFileDirectoryEntry(TagID, TiffType::LONG, 1), mValue(aValue)
{
}

uint IFDImageLength::Value() const
{
    return mValue;
}

void IFDImageLength::SavePass1(std::ostream &aStream)
{
    aStream.write(reinterpret_cast<char *>(&mTag.TagID), sizeof(mTag.TagID));
    TiffType temp = TiffType::LONG;
    aStream.write(reinterpret_cast<char *>(&temp), sizeof(temp));
    int count = 1;
    aStream.write(reinterpret_cast<char *>(&count), sizeof(count));
    aStream.write(reinterpret_cast<char *>(&mValue), sizeof(mValue));
}

IFDImageWidth::IFDImageWidth(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), ImageFileDirectoryEntry(aFile, aTagID)
{
    if (GetTiffTypeSizeInBytes(mTag.Type) == 2 && !IsLittleEndian())
    {
        uint *ptr     = &mTag.Offset.UIntVal; // NOLINT
        ushort *ptrUS = reinterpret_cast<ushort *>(ptr);
        mValue        = ptrUS[4 / GetTiffTypeSizeInBytes(mTag.Type) - 1];
    }
    else
    {
        mValue = mTag.Offset.UIntVal; // NOLINT
    }
}

IFDImageWidth::IFDImageWidth(uint aValue)
    : File("", true), ImageFileDirectoryEntry(TagID, TiffType::LONG, 1), mValue(aValue)
{
}

uint IFDImageWidth::Value() const
{
    return mValue;
}

void IFDImageWidth::SavePass1(std::ostream &aStream)
{
    aStream.write(reinterpret_cast<char *>(&mTag.TagID), sizeof(mTag.TagID));
    TiffType temp = TiffType::LONG;
    aStream.write(reinterpret_cast<char *>(&temp), sizeof(temp));
    int count = 1;
    aStream.write(reinterpret_cast<char *>(&count), sizeof(count));
    aStream.write(reinterpret_cast<char *>(&mValue), sizeof(mValue));
}

IFDRowsPerStrip::IFDRowsPerStrip(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), ImageFileDirectoryEntry(aFile, aTagID)
{
    if (GetTiffTypeSizeInBytes(mTag.Type) == 2 && !aFile.IsLittleEndian())
    {
        uint *ptr     = &mTag.Offset.UIntVal; // NOLINT
        ushort *ptrUS = reinterpret_cast<ushort *>(ptr);
        mValue        = ptrUS[4 / GetTiffTypeSizeInBytes(mTag.Type) - 1];
    }
    else
    {
        mValue = mTag.Offset.UIntVal; // NOLINT
    }
}

IFDRowsPerStrip::IFDRowsPerStrip(uint aValue)
    : File("", true), ImageFileDirectoryEntry(TagID, TiffType::LONG, 1), mValue(aValue)
{
}

uint IFDRowsPerStrip::Value() const
{
    return mValue;
}

void IFDRowsPerStrip::SavePass1(std::ostream &aStream)
{
    aStream.write(reinterpret_cast<char *>(&mTag.TagID), sizeof(mTag.TagID));
    TiffType temp = TiffType::LONG;
    aStream.write(reinterpret_cast<char *>(&temp), sizeof(temp));
    int count = 1;
    aStream.write(reinterpret_cast<char *>(&count), sizeof(count));
    aStream.write(reinterpret_cast<char *>(&mValue), sizeof(mValue));
}

IFDStripByteCounts::IFDStripByteCounts(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), ImageFileDirectoryEntry(aFile, aTagID)
{
    if (GetTiffTypeSizeInBytes(mTag.Type) == 2)
    {
        if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
        {
            uint *ptr     = &mTag.Offset.UIntVal; // NOLINT
            ushort *ptrUS = reinterpret_cast<ushort *>(ptr);
            mValue.resize(mTag.Count);
            for (uint i = 0; i < mTag.Count; i++)
            {
                if (!IsLittleEndian())
                {
                    mValue[i] = ptrUS[4 / GetTiffTypeSizeInBytes(mTag.Type) - i - 1];
                }
                else
                {
                    mValue[i] = ptrUS[i];
                }
            }
        }
        else
        {
            const size_t currentOffset = TellRead();
            SeekRead(mTag.Offset.UIntVal, std::ios_base::beg); // NOLINT

            std::vector<ushort> temp = ReadLE<ushort>(mTag.Count);
            mValue.resize(mTag.Count);

            std::copy(temp.begin(), temp.end(), mValue.begin());

            SeekRead(currentOffset, std::ios_base::beg);
        }
    }
    else
    {
        if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
        {
            mValue.resize(mTag.Count);
            mValue[0] = mTag.Offset.UIntVal; // NOLINT
        }
        else
        {
            const size_t currentOffset = TellRead();
            SeekRead(mTag.Offset.UIntVal, std::ios_base::beg); // NOLINT

            mValue = ReadLE<uint>(mTag.Count);
            SeekRead(currentOffset, std::ios_base::beg);
        }
    }
}

IFDStripByteCounts::IFDStripByteCounts(std::vector<uint> &&aValues)
    : File("", true), ImageFileDirectoryEntry(TagID, TiffType::LONG, to_uint(aValues.size())),
      mValue(std::move(aValues))
{
}

IFDStripByteCounts::IFDStripByteCounts(uint aValue) : File("", true), ImageFileDirectoryEntry(TagID, TiffType::LONG, 1)
{
    mValue.push_back(aValue);
}

const std::vector<uint> &IFDStripByteCounts::Value() const
{
    return mValue;
}

void IFDStripByteCounts::SavePass1(std::ostream &aStream)
{
    if (mValue.size() == 1)
    {
        char temp[sizeof(uint)] = {0};
        std::memcpy(reinterpret_cast<char *>(temp), mValue.data(), sizeof(uint));
        const uint val = *(reinterpret_cast<uint *>(temp));

        WriteEntryHeader(val, aStream, to_int(mValue.size()));
        mOffsetInStream = 0;
    }
    else
    {
        mOffsetInStream = WriteEntryHeader(0, aStream, to_int(mValue.size()));
    }
}

void IFDStripByteCounts::SavePass2(std::ostream &aStream)
{
    if (mOffsetInStream != 0)
    {
        WritePass2(reinterpret_cast<char *>(mValue.data()), mValue.size() * sizeof(uint), aStream);
    }
}

IFDStripOffsets::IFDStripOffsets(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), ImageFileDirectoryEntry(aFile, aTagID)
{
    if (GetTiffTypeSizeInBytes(mTag.Type) == 2)
    {
        if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
        {
            uint *ptr     = &mTag.Offset.UIntVal; // NOLINT
            ushort *ptrUS = reinterpret_cast<ushort *>(ptr);
            mValue.resize(mTag.Count);
            for (uint i = 0; i < mTag.Count; i++)
            {
                if (!IsLittleEndian())
                {
                    mValue[i] = ptrUS[4 / GetTiffTypeSizeInBytes(mTag.Type) - i - 1];
                }
                else
                {
                    mValue[i] = ptrUS[i];
                }
            }
        }
        else
        {
            const size_t currentOffset = TellRead();
            SeekRead(mTag.Offset.UIntVal, std::ios_base::beg); // NOLINT

            std::vector<ushort> temp = ReadLE<ushort>(mTag.Count);
            mValue.resize(mTag.Count);

            std::copy(temp.begin(), temp.end(), mValue.begin());

            SeekRead(currentOffset, std::ios_base::beg);
        }
    }
    else
    {
        if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 4)
        {
            mValue.resize(mTag.Count);
            mValue[0] = mTag.Offset.UIntVal; // NOLINT
        }
        else
        {
            const size_t currentOffset = TellRead();
            SeekRead(mTag.Offset.UIntVal, std::ios_base::beg); // NOLINT

            mValue = ReadLE<uint>(mTag.Count);
            SeekRead(currentOffset, std::ios_base::beg);
        }
    }
}

IFDStripOffsets::IFDStripOffsets(size_t aStripCount)
    : File("", true), ImageFileDirectoryEntry(TagID, TiffType::LONG, 1), mValue(aStripCount, 0)
{
}

const std::vector<uint> &IFDStripOffsets::Value() const
{
    return mValue;
}

void IFDStripOffsets::SavePass1(std::ostream &aStream)
{
    if (mValue.size() == 1)
    {
        char temp[4] = {0};
        std::memcpy(reinterpret_cast<char *>(temp), mValue.data(), sizeof(uint) * mValue.size());
        const uint val = *(reinterpret_cast<uint *>(temp));

        // special case as these values get written in the third and last pass
        mOffsetInStream = WriteEntryHeader(val, aStream, to_int(mValue.size()));
    }
    else
    {
        mOffsetInStream = WriteEntryHeader(0, aStream, to_int(mValue.size()));
    }
}

void IFDStripOffsets::SavePass2(std::ostream &aStream)
{
    if (mValue.size() == 1)
    {
        return;
    }
    if (mOffsetInStream != 0)
    {
        // write dummy data to final destination:
        aStream.seekp(0, std::ios_base::end);
        uint offset = to_uint(size_t(aStream.tellp()));
        offset += offset % 4; // align offsets to largest word boundary
        aStream.seekp(offset, std::ios_base::beg);

        aStream.write(reinterpret_cast<char *>(mValue.data()), std::streamsize(mValue.size() * sizeof(uint)));

        // store current offset in TAG:
        aStream.seekp(std::streamoff(mOffsetInStream), std::ios_base::beg);
        aStream.write(reinterpret_cast<char *>(&offset), sizeof(offset));

        // memorize where to write the final offsets to:
        mOffsetInStream = offset;

        // goto the end of the file again:
        aStream.seekp(0, std::ios_base::end);
    }
}

void IFDStripOffsets::SaveFinalOffsets(std::ostream &aStream, std::vector<uint> &aFinalOffsets)
{
    aStream.seekp(std::streamoff(mOffsetInStream), std::ios_base::beg);

    aStream.write(reinterpret_cast<char *>(aFinalOffsets.data()), std::streamsize(aFinalOffsets.size() * sizeof(uint)));

    aStream.seekp(0, std::ios_base::end);
}

IFDArtist::IFDArtist(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<std::string>(aFile, aTagID)
{
}

IFDArtist::IFDArtist(const std::string &aValue) : File("", true), IFDEntry<std::string>(aValue, TagID)
{
}

IFDCopyright::IFDCopyright(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<std::string>(aFile, aTagID)
{
}

IFDCopyright::IFDCopyright(const std::string &aValue) : File("", true), IFDEntry<std::string>(aValue, TagID)
{
}

IFDDateTime::IFDDateTime(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<std::string>(aFile, aTagID)
{
}

IFDDateTime::IFDDateTime(const std::string &aValue) : File("", true), IFDEntry<std::string>(aValue, TagID)
{
}

IFDHostComputer::IFDHostComputer(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<std::string>(aFile, aTagID)
{
}

IFDHostComputer::IFDHostComputer(const std::string &aValue) : File("", true), IFDEntry<std::string>(aValue, TagID)
{
}

IFDImageDescription::IFDImageDescription(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<std::string>(aFile, aTagID)
{
}

IFDImageDescription::IFDImageDescription(const std::string &aValue)
    : File("", true), IFDEntry<std::string>(aValue, TagID)
{
}

IFDModel::IFDModel(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<std::string>(aFile, aTagID)
{
}

IFDModel::IFDModel(const std::string &aValue) : File("", true), IFDEntry<std::string>(aValue, TagID)
{
}

IFDMake::IFDMake(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<std::string>(aFile, aTagID)
{
}

IFDMake::IFDMake(const std::string &aValue) : File("", true), IFDEntry<std::string>(aValue, TagID)
{
}

IFDSoftware::IFDSoftware(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<std::string>(aFile, aTagID)
{
}

IFDSoftware::IFDSoftware(const std::string &aValue) : File("", true), IFDEntry<std::string>(aValue, TagID)
{
}

IFDBitsPerSample::IFDBitsPerSample(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<ushort>(aFile, aTagID)
{
}

IFDBitsPerSample::IFDBitsPerSample(ushort aValue) : File("", true), IFDEntry<ushort>(aValue, TagID, TiffType::SHORT)
{
}

IFDBitsPerSample::IFDBitsPerSample(std::vector<ushort> &aValue)
    : File("", true), IFDEntry<ushort>(std::move(aValue), TagID, TiffType::SHORT)
{
}

ushort IFDBitsPerSample::Value(size_t aIdx) const
{
    if (aIdx >= mValue.size())
    {
        return 0;
    }
    return mValue[aIdx];
}

IFDCellLength::IFDCellLength(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<ushort>(aFile, aTagID)
{
}

IFDCellLength::IFDCellLength(ushort aValue) : File("", true), IFDEntry<ushort>(aValue, TagID, TiffType::SHORT)
{
}

ushort IFDCellLength::Value() const
{
    return mValue[0];
}

IFDCellWidth::IFDCellWidth(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<ushort>(aFile, aTagID)
{
}

IFDCellWidth::IFDCellWidth(ushort aValue) : File("", true), IFDEntry<ushort>(aValue, TagID, TiffType::SHORT)
{
}

ushort IFDCellWidth::Value() const
{
    return mValue[0];
}

IFDColorMap::IFDColorMap(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<ushort>(aFile, aTagID)
{
}

IFDColorMap::IFDColorMap(std::vector<ushort> &aValues)
    : File("", true), IFDEntry<ushort>(std::move(aValues), TagID, TiffType::SHORT)
{
}

const std::vector<ushort> &IFDColorMap::Value() const
{
    return mValue;
}

IFDCompression::IFDCompression(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<TIFFCompression>(aFile, aTagID)
{
}

IFDCompression::IFDCompression(TIFFCompression aValue)
    : File("", true), IFDEntry<TIFFCompression>(aValue, TagID, TiffType::SHORT)
{
}

TIFFCompression IFDCompression::Value() const
{
    return mValue[0];
}

IFDExtraSamples::IFDExtraSamples(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<ushort>(aFile, aTagID)
{
}

IFDExtraSamples::IFDExtraSamples(ushort aValue) : File("", true), IFDEntry<ushort>(aValue, TagID, TiffType::SHORT)
{
}

ushort IFDExtraSamples::Value() const
{
    return mValue[0];
}

IFDDifferencingPredictor::IFDDifferencingPredictor(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<TIFFDifferencingPredictor>(aFile, aTagID)
{
}

IFDDifferencingPredictor::IFDDifferencingPredictor(TIFFDifferencingPredictor aValue)
    : File("", true), IFDEntry<TIFFDifferencingPredictor>(aValue, TagID, TiffType::SHORT)
{
}

TIFFDifferencingPredictor IFDDifferencingPredictor::Value() const
{
    return mValue[0];
}

IFDFillOrder::IFDFillOrder(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<ushort>(aFile, aTagID)
{
}

IFDFillOrder::IFDFillOrder(ushort aValue) : File("", true), IFDEntry<ushort>(aValue, TagID, TiffType::SHORT)
{
}

ushort IFDFillOrder::Value() const
{
    return mValue[0];
}

IFDFreeByteCounts::IFDFreeByteCounts(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<uint>(aFile, aTagID)
{
}

IFDFreeByteCounts::IFDFreeByteCounts(uint aValue) : File("", true), IFDEntry<uint>(aValue, TagID, TiffType::LONG)
{
}

uint IFDFreeByteCounts::Value() const
{
    return mValue[0];
}

IFDFreeOffsets::IFDFreeOffsets(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<uint>(aFile, aTagID)
{
}

IFDFreeOffsets::IFDFreeOffsets(uint aValue) : File("", true), IFDEntry<uint>(aValue, TagID, TiffType::LONG)
{
}

uint IFDFreeOffsets::Value() const
{
    return mValue[0];
}

IFDGrayResponseCurve::IFDGrayResponseCurve(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<ushort>(aFile, aTagID)
{
}

IFDGrayResponseCurve::IFDGrayResponseCurve(std::vector<ushort> &aValues)
    : File("", true), IFDEntry<ushort>(std::move(aValues), TagID, TiffType::SHORT)
{
}

const std::vector<ushort> &IFDGrayResponseCurve::Value() const
{
    return mValue;
}

IFDGrayResponseUnit::IFDGrayResponseUnit(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<ushort>(aFile, aTagID)
{
}

IFDGrayResponseUnit::IFDGrayResponseUnit(ushort aValue)
    : File("", true), IFDEntry<ushort>(aValue, TagID, TiffType::SHORT)
{
}

ushort IFDGrayResponseUnit::Value() const
{
    return mValue[0];
}

IFDMaxSampleValue::IFDMaxSampleValue(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<ushort>(aFile, aTagID)
{
}

IFDMaxSampleValue::IFDMaxSampleValue(ushort aValue) : File("", true), IFDEntry<ushort>(aValue, TagID, TiffType::SHORT)
{
}

ushort IFDMaxSampleValue::Value() const
{
    return mValue[0];
}

IFDMinSampleValue::IFDMinSampleValue(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<ushort>(aFile, aTagID)
{
}

IFDMinSampleValue::IFDMinSampleValue(ushort aValue) : File("", true), IFDEntry<ushort>(aValue, TagID, TiffType::SHORT)
{
}

ushort IFDMinSampleValue::Value() const
{
    return mValue[0];
}

IFDNewSubfileType::IFDNewSubfileType(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<uint>(aFile, aTagID)
{
}

IFDNewSubfileType::IFDNewSubfileType(uint aValue) : File("", true), IFDEntry<uint>(aValue, TagID, TiffType::LONG)
{
}

uint IFDNewSubfileType::Value() const
{
    return mValue[0];
}

IFDOrientation::IFDOrientation(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<TiffOrientation>(aFile, aTagID)
{
}

IFDOrientation::IFDOrientation(TiffOrientation aValue)
    : File("", true), IFDEntry<TiffOrientation>(aValue, TagID, TiffType::SHORT)
{
}

TiffOrientation IFDOrientation::Value() const
{
    return mValue[0];
}

IFDPhotometricInterpretation::IFDPhotometricInterpretation(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<TIFFPhotometricInterpretation>(aFile, aTagID)
{
}

IFDPhotometricInterpretation::IFDPhotometricInterpretation(TIFFPhotometricInterpretation aValue)
    : File("", true), IFDEntry<TIFFPhotometricInterpretation>(aValue, TagID, TiffType::SHORT)
{
}

TIFFPhotometricInterpretation IFDPhotometricInterpretation::Value() const
{
    return mValue[0];
}

IFDPlanarConfiguration::IFDPlanarConfiguration(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<TIFFPlanarConfigurartion>(aFile, aTagID)
{
}

IFDPlanarConfiguration::IFDPlanarConfiguration(TIFFPlanarConfigurartion aValue)
    : File("", true), IFDEntry<TIFFPlanarConfigurartion>(aValue, TagID, TiffType::SHORT)
{
}

TIFFPlanarConfigurartion IFDPlanarConfiguration::Value() const
{
    return mValue[0];
}

IFDResolutionUnit::IFDResolutionUnit(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<TIFFResolutionUnit>(aFile, aTagID)
{
}

IFDResolutionUnit::IFDResolutionUnit(TIFFResolutionUnit aValue)
    : File("", true), IFDEntry<TIFFResolutionUnit>(aValue, TagID, TiffType::SHORT)
{
}

TIFFResolutionUnit IFDResolutionUnit::Value() const
{
    return mValue[0];
}

IFDSamplesPerPixel::IFDSamplesPerPixel(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<ushort>(aFile, aTagID)
{
}

IFDSamplesPerPixel::IFDSamplesPerPixel(ushort aValue) : File("", true), IFDEntry<ushort>(aValue, TagID, TiffType::SHORT)
{
}

ushort IFDSamplesPerPixel::Value() const
{
    return mValue[0];
}

IFDSampleFormat::IFDSampleFormat(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<TIFFSampleFormat>(aFile, aTagID)
{
}

IFDSampleFormat::IFDSampleFormat(TIFFSampleFormat aValue)
    : File("", true), IFDEntry<TIFFSampleFormat>(aValue, TagID, TiffType::SHORT)
{
}

TIFFSampleFormat IFDSampleFormat::Value() const
{
    return mValue[0];
}

IFDSubfileType::IFDSubfileType(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<ushort>(aFile, aTagID)
{
}

IFDSubfileType::IFDSubfileType(ushort aValue) : File("", true), IFDEntry<ushort>(aValue, TagID, TiffType::SHORT)
{
}

ushort IFDSubfileType::Value() const
{
    return mValue[0];
}

IFDThreshholding::IFDThreshholding(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<ushort>(aFile, aTagID)
{
}

IFDThreshholding::IFDThreshholding(ushort aValue) : File("", true), IFDEntry<ushort>(aValue, TagID, TiffType::SHORT)
{
}

ushort IFDThreshholding::Value() const
{
    return mValue[0];
}

IFDXResolution::IFDXResolution(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<Rational>(aFile, aTagID)
{
    // if endianess was inversed, denominator and nominator are now inverted, too
    if (!aFile.IsLittleEndian())
    {
        const uint temp       = mValue[0].denominator;
        mValue[0].denominator = mValue[0].nominator;
        mValue[0].nominator   = temp;
    }
}

IFDXResolution::IFDXResolution(Rational aValue) : File("", true), IFDEntry<Rational>(aValue, TagID, TiffType::RATIONAL)
{
}

Rational IFDXResolution::Value() const
{
    return mValue[0];
}

IFDYResolution::IFDYResolution(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), IFDEntry<Rational>(aFile, aTagID)
{
    // if endianess was inversed, denominator and nominator are now inverted, too
    if (!aFile.IsLittleEndian())
    {
        const uint temp       = mValue[0].denominator;
        mValue[0].denominator = mValue[0].nominator;
        mValue[0].nominator   = temp;
    }
}

IFDYResolution::IFDYResolution(Rational aValue) : File("", true), IFDEntry<Rational>(aValue, TagID, TiffType::RATIONAL)
{
}

Rational IFDYResolution::Value() const
{
    return mValue[0];
}

} // namespace opp::fileIO::tiffTag