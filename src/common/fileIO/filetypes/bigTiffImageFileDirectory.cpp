// --> tiffFile.h must be included first
// clang-format off
#include "bigTiffImageFileDirectory.h"
#include "bigTiffImageFileDirectory_impl.h" //NOLINT
#include "tiffFile.h"
// clang-format on

#include "tiffImageFileDirectory.h"
#include <algorithm>
#include <common/defines.h>
#include <common/fileIO/file.h>
#include <common/fileIO/filetypes/tiffImageFileDirectory.h>
#include <common/fileIO/pseudoFileReader.h>
#include <cstddef>
#include <ios>
#include <memory>
#include <string>
#include <vector>

namespace opp::fileIO
{
using namespace tiffTag;
namespace bigTiffTag
{

BigImageFileDirectory::BigImageFileDirectory(TIFFFile &aFile) : mTifffile(aFile)
{
    mEntryCount = mTifffile.ReadLE<ulong64>(); // NOLINT(cppcoreguidelines-prefer-member-initializer)
    for (ulong64 i = 0; i < mEntryCount; i++)
    {
        mEntries.push_back(BigImageFileDirectoryEntry::CreateFileDirectoryEntry(mTifffile));
    }
}

std::shared_ptr<BigImageFileDirectoryEntry> BigImageFileDirectory::GetEntry(ushort aTagID)
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

BigImageFileDirectoryEntry::BigImageFileDirectoryEntry(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), PseudoFileReader(aFile.ReadStream()), mTag()
{
    mTag.TagID            = aTagID;
    mTag.Type             = ReadLE<TiffType>();
    mTag.Count            = ReadLE<ulong64>();
    mTag.Offset.UInt64Val = ReadLE<ulong64>(); // NOLINT
}

BigImageFileDirectoryEntry::BigImageFileDirectoryEntry(TIFFFile &aFile)
    : File("", aFile.IsLittleEndian()), PseudoFileReader(aFile.ReadStream()), mTag()
{
    mTag.TagID            = ReadLE<ushort>();
    mTag.Type             = ReadLE<TiffType>();
    mTag.Count            = ReadLE<ulong64>();
    mTag.Offset.UInt64Val = ReadLE<ulong64>(); // NOLINT
}

std::shared_ptr<BigImageFileDirectoryEntry> BigImageFileDirectoryEntry::CreateFileDirectoryEntry(TIFFFile &aFile)
{
    const ushort tagID = aFile.ReadLE<ushort>();
    switch (tagID)
    {
        case BigIFDArtist::TagID:
            return std::make_shared<BigIFDArtist>(aFile, tagID);
        case BigIFDBitsPerSample::TagID:
            return std::make_shared<BigIFDBitsPerSample>(aFile, tagID);
        case BigIFDCellLength::TagID:
            return std::make_shared<BigIFDCellLength>(aFile, tagID);
        case BigIFDCellWidth::TagID:
            return std::make_shared<BigIFDCellWidth>(aFile, tagID);
        case BigIFDColorMap::TagID:
            return std::make_shared<BigIFDColorMap>(aFile, tagID);
        case BigIFDCompression::TagID:
            return std::make_shared<BigIFDCompression>(aFile, tagID);
        case BigIFDCopyright::TagID:
            return std::make_shared<BigIFDCopyright>(aFile, tagID);
        case BigIFDDateTime::TagID:
            return std::make_shared<BigIFDDateTime>(aFile, tagID);
        case BigIFDExtraSamples::TagID:
            return std::make_shared<BigIFDExtraSamples>(aFile, tagID);
        case BigIFDDifferencingPredictor::TagID:
            return std::make_shared<BigIFDDifferencingPredictor>(aFile, tagID);
        case BigIFDFillOrder::TagID:
            return std::make_shared<BigIFDFillOrder>(aFile, tagID);
        case BigIFDFreeByteCounts::TagID:
            return std::make_shared<BigIFDFreeByteCounts>(aFile, tagID);
        case BigIFDFreeOffsets::TagID:
            return std::make_shared<BigIFDFreeOffsets>(aFile, tagID);
        case BigIFDGrayResponseCurve::TagID:
            return std::make_shared<BigIFDGrayResponseCurve>(aFile, tagID);
        case BigIFDGrayResponseUnit::TagID:
            return std::make_shared<BigIFDGrayResponseUnit>(aFile, tagID);
        case BigIFDHostComputer::TagID:
            return std::make_shared<BigIFDHostComputer>(aFile, tagID);
        case BigIFDImageDescription::TagID:
            return std::make_shared<BigIFDImageDescription>(aFile, tagID);
        case BigIFDImageLength::TagID:
            return std::make_shared<BigIFDImageLength>(aFile, tagID);
        case BigIFDImageWidth::TagID:
            return std::make_shared<BigIFDImageWidth>(aFile, tagID);
        case BigIFDMake::TagID:
            return std::make_shared<BigIFDMake>(aFile, tagID);
        case BigIFDMaxSampleValue::TagID:
            return std::make_shared<BigIFDMaxSampleValue>(aFile, tagID);
        case BigIFDMinSampleValue::TagID:
            return std::make_shared<BigIFDMinSampleValue>(aFile, tagID);
        case BigIFDModel::TagID:
            return std::make_shared<BigIFDModel>(aFile, tagID);
        case BigIFDNewSubfileType::TagID:
            return std::make_shared<BigIFDNewSubfileType>(aFile, tagID);
        case BigIFDOrientation::TagID:
            return std::make_shared<BigIFDOrientation>(aFile, tagID);
        case BigIFDPhotometricInterpretation::TagID:
            return std::make_shared<BigIFDPhotometricInterpretation>(aFile, tagID);
        case BigIFDPlanarConfiguration::TagID:
            return std::make_shared<BigIFDPlanarConfiguration>(aFile, tagID);
        case BigIFDResolutionUnit::TagID:
            return std::make_shared<BigIFDResolutionUnit>(aFile, tagID);
        case BigIFDRowsPerStrip::TagID:
            return std::make_shared<BigIFDRowsPerStrip>(aFile, tagID);
        case BigIFDSamplesPerPixel::TagID:
            return std::make_shared<BigIFDSamplesPerPixel>(aFile, tagID);
        case BigIFDSampleFormat::TagID:
            return std::make_shared<BigIFDSampleFormat>(aFile, tagID);
        case BigIFDSoftware::TagID:
            return std::make_shared<BigIFDSoftware>(aFile, tagID);
        case BigIFDStripByteCounts::TagID:
            return std::make_shared<BigIFDStripByteCounts>(aFile, tagID);
        case BigIFDStripOffsets::TagID:
            return std::make_shared<BigIFDStripOffsets>(aFile, tagID);
        case BigIFDSubfileType::TagID:
            return std::make_shared<BigIFDSubfileType>(aFile, tagID);
        case BigIFDThreshholding::TagID:
            return std::make_shared<BigIFDThreshholding>(aFile, tagID);
        case BigIFDXResolution::TagID:
            return std::make_shared<BigIFDXResolution>(aFile, tagID);
        case BigIFDYResolution::TagID:
            return std::make_shared<BigIFDYResolution>(aFile, tagID);
        default:
            return std::shared_ptr<BigImageFileDirectoryEntry>(new BigImageFileDirectoryEntry(aFile, tagID));
    }
}

BigIFDImageLength::BigIFDImageLength(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigImageFileDirectoryEntry(aFile, aTagID)
{
    if (GetTiffTypeSizeInBytes(mTag.Type) == 2 && !IsLittleEndian())
    {
        ulong64 *ptr  = &mTag.Offset.UInt64Val; // NOLINT
        ushort *ptrUS = reinterpret_cast<ushort *>(ptr);
        mValue        = ptrUS[8 / GetTiffTypeSizeInBytes(mTag.Type) - 1];
    }
    else if (GetTiffTypeSizeInBytes(mTag.Type) == 4 && !IsLittleEndian())
    {
        ulong64 *ptr = &mTag.Offset.UInt64Val; // NOLINT
        uint *ptrUI  = reinterpret_cast<uint *>(ptr);
        mValue       = ptrUI[8 / GetTiffTypeSizeInBytes(mTag.Type) - 1];
    }
    else
    {
        mValue = mTag.Offset.UIntVal; // NOLINT
    }
}

uint BigIFDImageLength::Value() const
{
    return mValue;
}

BigIFDImageWidth::BigIFDImageWidth(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigImageFileDirectoryEntry(aFile, aTagID)
{
    if (GetTiffTypeSizeInBytes(mTag.Type) == 2 && !IsLittleEndian())
    {
        ulong64 *ptr  = &mTag.Offset.UInt64Val; // NOLINT
        ushort *ptrUS = reinterpret_cast<ushort *>(ptr);
        mValue        = ptrUS[8 / GetTiffTypeSizeInBytes(mTag.Type) - 1];
    }
    else if (GetTiffTypeSizeInBytes(mTag.Type) == 4 && !IsLittleEndian())
    {
        ulong64 *ptr = &mTag.Offset.UInt64Val; // NOLINT
        uint *ptrUI  = reinterpret_cast<uint *>(ptr);
        mValue       = ptrUI[8 / GetTiffTypeSizeInBytes(mTag.Type) - 1];
    }
    else
    {
        mValue = mTag.Offset.UIntVal; // NOLINT
    }
}

uint BigIFDImageWidth::Value() const
{
    return mValue;
}

BigIFDRowsPerStrip::BigIFDRowsPerStrip(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigImageFileDirectoryEntry(aFile, aTagID)
{
    if (GetTiffTypeSizeInBytes(mTag.Type) == 2 && !IsLittleEndian())
    {
        ulong64 *ptr  = &mTag.Offset.UInt64Val; // NOLINT
        ushort *ptrUS = reinterpret_cast<ushort *>(ptr);
        mValue        = ptrUS[8 / GetTiffTypeSizeInBytes(mTag.Type) - 1];
    }
    else if (GetTiffTypeSizeInBytes(mTag.Type) == 4 && !IsLittleEndian())
    {
        ulong64 *ptr = &mTag.Offset.UInt64Val; // NOLINT
        uint *ptrUI  = reinterpret_cast<uint *>(ptr);
        mValue       = ptrUI[8 / GetTiffTypeSizeInBytes(mTag.Type) - 1];
    }
    else
    {
        mValue = mTag.Offset.UIntVal; // NOLINT
    }
}

uint BigIFDRowsPerStrip::Value() const
{
    return mValue;
}

BigIFDStripByteCounts::BigIFDStripByteCounts(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigImageFileDirectoryEntry(aFile, aTagID)
{
    if (GetTiffTypeSizeInBytes(mTag.Type) == 2)
    {
        if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 8)
        {
            ulong64 *ptr  = &mTag.Offset.UInt64Val; // NOLINT
            ushort *ptrUS = reinterpret_cast<ushort *>(ptr);
            mValue.resize(mTag.Count);
            for (uint i = 0; i < mTag.Count; i++)
            {
                if (!IsLittleEndian())
                {
                    mValue[i] = ptrUS[8 / GetTiffTypeSizeInBytes(mTag.Type) - i - 1];
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
            SeekRead(mTag.Offset.UInt64Val, std::ios_base::beg); // NOLINT

            std::vector<ushort> temp = ReadLE<ushort>(mTag.Count);
            mValue.resize(mTag.Count);
            std::copy(temp.begin(), temp.end(), mValue.begin());
            SeekRead(currentOffset, std::ios_base::beg);
        }
    }
    else if (GetTiffTypeSizeInBytes(mTag.Type) == 4)
    {
        if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 8)
        {
            ulong64 *ptr = &mTag.Offset.UInt64Val; // NOLINT
            auto *ptrUI  = reinterpret_cast<uint *>(ptr);
            mValue.resize(mTag.Count);
            for (uint i = 0; i < mTag.Count; i++)
            {
                if (!IsLittleEndian())
                {
                    mValue[i] = ptrUI[8 / GetTiffTypeSizeInBytes(mTag.Type) - i - 1];
                }
                else
                {
                    mValue[i] = ptrUI[i];
                }
            }
        }
        else
        {
            const size_t currentOffset = TellRead();
            SeekRead(mTag.Offset.UInt64Val, std::ios_base::beg); // NOLINT

            std::vector<uint> temp = ReadLE<uint>(mTag.Count);
            mValue.resize(mTag.Count);
            std::copy(temp.begin(), temp.end(), mValue.begin());

            SeekRead(currentOffset, std::ios_base::beg);
        }
    }
    else
    {
        if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 8)
        {
            mValue.resize(mTag.Count);
            mValue[0] = mTag.Offset.UInt64Val; // NOLINT
        }
        else
        {
            const size_t currentOffset = TellRead();
            SeekRead(mTag.Offset.UInt64Val, std::ios_base::beg); // NOLINT

            mValue = ReadLE<ulong64>(mTag.Count);

            SeekRead(currentOffset, std::ios_base::beg);
        }
    }
}

const std::vector<ulong64> &BigIFDStripByteCounts::Value() const
{
    return mValue;
}

BigIFDStripOffsets::BigIFDStripOffsets(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigImageFileDirectoryEntry(aFile, aTagID)
{
    if (GetTiffTypeSizeInBytes(mTag.Type) == 2)
    {
        if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 8)
        {
            ulong64 *ptr  = &mTag.Offset.UInt64Val; // NOLINT
            ushort *ptrUS = reinterpret_cast<ushort *>(ptr);
            mValue.resize(mTag.Count);
            for (uint i = 0; i < mTag.Count; i++)
            {
                if (!IsLittleEndian())
                {
                    mValue[i] = ptrUS[8 / GetTiffTypeSizeInBytes(mTag.Type) - i - 1];
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
            SeekRead(mTag.Offset.UInt64Val, std::ios_base::beg); // NOLINT

            std::vector<ushort> temp = ReadLE<ushort>(mTag.Count);
            mValue.resize(mTag.Count);
            std::copy(temp.begin(), temp.end(), mValue.begin());
            SeekRead(currentOffset, std::ios_base::beg);
        }
    }
    else if (GetTiffTypeSizeInBytes(mTag.Type) == 4)
    {
        if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 8)
        {
            ulong64 *ptr = &mTag.Offset.UInt64Val; // NOLINT
            uint *ptrUI  = reinterpret_cast<uint *>(ptr);
            mValue.resize(mTag.Count);
            for (uint i = 0; i < mTag.Count; i++)
            {
                if (!IsLittleEndian())
                {
                    mValue[i] = ptrUI[8 / GetTiffTypeSizeInBytes(mTag.Type) - i - 1];
                }
                else
                {
                    mValue[i] = ptrUI[i];
                }
            }
        }
        else
        {
            const size_t currentOffset = TellRead();
            SeekRead(mTag.Offset.UInt64Val, std::ios_base::beg); // NOLINT

            std::vector<uint> temp = ReadLE<uint>(mTag.Count);
            mValue.resize(mTag.Count);
            std::copy(temp.begin(), temp.end(), mValue.begin());

            SeekRead(currentOffset, std::ios_base::beg);
        }
    }
    else
    {
        if (GetTiffTypeSizeInBytes(mTag.Type) * mTag.Count <= 8)
        {
            mValue.resize(mTag.Count);
            mValue[0] = mTag.Offset.UInt64Val; // NOLINT
        }
        else
        {
            const size_t currentOffset = TellRead();
            SeekRead(mTag.Offset.UInt64Val, std::ios_base::beg); // NOLINT

            mValue = ReadLE<ulong64>(mTag.Count);

            SeekRead(currentOffset, std::ios_base::beg);
        }
    }
}

const std::vector<ulong64> &BigIFDStripOffsets::Value() const
{
    return mValue;
}

BigIFDArtist::BigIFDArtist(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<std::string>(aFile, aTagID)
{
}

BigIFDCopyright::BigIFDCopyright(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<std::string>(aFile, aTagID)
{
}

BigIFDDateTime::BigIFDDateTime(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<std::string>(aFile, aTagID)
{
}

BigIFDHostComputer::BigIFDHostComputer(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<std::string>(aFile, aTagID)
{
}

BigIFDImageDescription::BigIFDImageDescription(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<std::string>(aFile, aTagID)
{
}

BigIFDModel::BigIFDModel(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<std::string>(aFile, aTagID)
{
}

BigIFDMake::BigIFDMake(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<std::string>(aFile, aTagID)
{
}

BigIFDSoftware::BigIFDSoftware(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<std::string>(aFile, aTagID)
{
}

BigIFDBitsPerSample::BigIFDBitsPerSample(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<ushort>(aFile, aTagID)
{
}

ushort BigIFDBitsPerSample::Value(size_t aIdx) const
{
    if (aIdx >= mValue.size())
    {
        return 0;
    }
    return mValue[aIdx];
}

BigIFDCellLength::BigIFDCellLength(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<ushort>(aFile, aTagID)
{
}

ushort BigIFDCellLength::Value() const
{
    return mValue[0];
}

BigIFDCellWidth::BigIFDCellWidth(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<ushort>(aFile, aTagID)
{
}

ushort BigIFDCellWidth::Value() const
{
    return mValue[0];
}

BigIFDColorMap::BigIFDColorMap(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<ushort>(aFile, aTagID)
{
}

ushort BigIFDColorMap::Value() const
{
    return mValue[0];
}

BigIFDCompression::BigIFDCompression(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<TIFFCompression>(aFile, aTagID)
{
}

TIFFCompression BigIFDCompression::Value() const
{
    return mValue[0];
}

BigIFDExtraSamples::BigIFDExtraSamples(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<ushort>(aFile, aTagID)
{
}

ushort BigIFDExtraSamples::Value() const
{
    return mValue[0];
}

BigIFDDifferencingPredictor::BigIFDDifferencingPredictor(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<TIFFDifferencingPredictor>(aFile, aTagID)
{
}

TIFFDifferencingPredictor BigIFDDifferencingPredictor::Value() const
{
    return mValue[0];
}

BigIFDFillOrder::BigIFDFillOrder(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<ushort>(aFile, aTagID)
{
}

ushort BigIFDFillOrder::Value() const
{
    return mValue[0];
}

BigIFDFreeByteCounts::BigIFDFreeByteCounts(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<uint>(aFile, aTagID)
{
}

uint BigIFDFreeByteCounts::Value() const
{
    return mValue[0];
}

BigIFDFreeOffsets::BigIFDFreeOffsets(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<uint>(aFile, aTagID)
{
}

uint BigIFDFreeOffsets::Value() const
{
    return mValue[0];
}

BigIFDGrayResponseCurve::BigIFDGrayResponseCurve(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<ushort>(aFile, aTagID)
{
}

const std::vector<ushort> &BigIFDGrayResponseCurve::Value() const
{
    return mValue;
}

BigIFDGrayResponseUnit::BigIFDGrayResponseUnit(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<ushort>(aFile, aTagID)
{
}

ushort BigIFDGrayResponseUnit::Value() const
{
    return mValue[0];
}

BigIFDMaxSampleValue::BigIFDMaxSampleValue(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<ushort>(aFile, aTagID)
{
}

ushort BigIFDMaxSampleValue::Value() const
{
    return mValue[0];
}

BigIFDMinSampleValue::BigIFDMinSampleValue(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<ushort>(aFile, aTagID)
{
}

ushort BigIFDMinSampleValue::Value() const
{
    return mValue[0];
}

BigIFDNewSubfileType::BigIFDNewSubfileType(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<uint>(aFile, aTagID)
{
}

uint BigIFDNewSubfileType::Value() const
{
    return mValue[0];
}

BigIFDOrientation::BigIFDOrientation(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<TiffOrientation>(aFile, aTagID)
{
}

TiffOrientation BigIFDOrientation::Value() const
{
    return mValue[0];
}

BigIFDPhotometricInterpretation::BigIFDPhotometricInterpretation(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<TIFFPhotometricInterpretation>(aFile, aTagID)
{
}

TIFFPhotometricInterpretation BigIFDPhotometricInterpretation::Value() const
{
    return mValue[0];
}

BigIFDPlanarConfiguration::BigIFDPlanarConfiguration(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<TIFFPlanarConfigurartion>(aFile, aTagID)
{
}

TIFFPlanarConfigurartion BigIFDPlanarConfiguration::Value() const
{
    return mValue[0];
}

BigIFDResolutionUnit::BigIFDResolutionUnit(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<TIFFResolutionUnit>(aFile, aTagID)
{
}

TIFFResolutionUnit BigIFDResolutionUnit::Value() const
{
    return mValue[0];
}

BigIFDSamplesPerPixel::BigIFDSamplesPerPixel(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<ushort>(aFile, aTagID)
{
}

ushort BigIFDSamplesPerPixel::Value() const
{
    return mValue[0];
}

BigIFDSampleFormat::BigIFDSampleFormat(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<TIFFSampleFormat>(aFile, aTagID)
{
}

TIFFSampleFormat BigIFDSampleFormat::Value() const
{
    return mValue[0];
}

BigIFDSubfileType::BigIFDSubfileType(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<ushort>(aFile, aTagID)
{
}

ushort BigIFDSubfileType::Value() const
{
    return mValue[0];
}

BigIFDThreshholding::BigIFDThreshholding(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<ushort>(aFile, aTagID)
{
}

ushort BigIFDThreshholding::Value() const
{
    return mValue[0];
}

BigIFDXResolution::BigIFDXResolution(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<Rational>(aFile, aTagID)
{
    // if endianess was inversed, denominator and nominator are now inverted, too
    if (!aFile.IsLittleEndian())
    {
        const uint temp       = mValue[0].denominator;
        mValue[0].denominator = mValue[0].nominator;
        mValue[0].nominator   = temp;
    }
}

Rational BigIFDXResolution::Value() const
{
    return mValue[0];
}

BigIFDYResolution::BigIFDYResolution(TIFFFile &aFile, ushort aTagID)
    : File("", aFile.IsLittleEndian()), BigIFDEntry<Rational>(aFile, aTagID)
{
    // if endianess was inversed, denominator and nominator are now inverted, too
    if (!aFile.IsLittleEndian())
    {
        const uint temp       = mValue[0].denominator;
        mValue[0].denominator = mValue[0].nominator;
        mValue[0].nominator   = temp;
    }
}

Rational BigIFDYResolution::Value() const
{
    return mValue[0];
}

} // namespace bigTiffTag
} // namespace opp::fileIO