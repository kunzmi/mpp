#pragma once
#include "common/defines.h"
#include "tiffImageFileDirectory.h"
#include <common/fileIO/pseudoFileReader.h>
#include <common/image/size2D.h>
#include <common/vector3.h>
#include <filesystem>
#include <memory>
#include <vector>

namespace opp::fileIO::bigTiffTag
{
struct BigTiffTag
{
    ushort TagID;
    tiffTag::TiffType Type;
    ulong64 Count;
    union
    {
        char CharVal;
        byte UCharVal;
        short ShortVal;
        ushort UShortVal;
        int IntVal;
        uint UIntVal;
        ulong64 UInt64Val;
    } Offset;
};

class BigImageFileDirectoryEntry : public PseudoFileReader
{
  protected:
    BigTiffTag mTag; // NOLINT -> should be private

    BigImageFileDirectoryEntry(TIFFFile &aFile, ushort aTagID);

  public:
    explicit BigImageFileDirectoryEntry(TIFFFile &aFile);
    static std::shared_ptr<BigImageFileDirectoryEntry> CreateFileDirectoryEntry(TIFFFile &aFile);

    friend class BigImageFileDirectory;

    [[nodiscard]] ushort GetTagID() const
    {
        return mTag.TagID;
    }
};

template <typename T> class BigIFDEntry : public BigImageFileDirectoryEntry
{
  protected:
    std::vector<T> mValue; // NOLINT -> should be private

  public:
    BigIFDEntry(TIFFFile &aFile, ushort aTagID);
};

template <> class BigIFDEntry<std::string> : public BigImageFileDirectoryEntry
{
  protected:
    std::string mValue; // NOLINT -> should be private

  public:
    BigIFDEntry(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] const std::string &Value() const;
};

class BigImageFileDirectory
{
  private:
    TIFFFile &mTifffile; // NOLINT --> avoid reference
    ulong64 mEntryCount{0};
    std::vector<std::shared_ptr<BigImageFileDirectoryEntry>> mEntries;

  public:
    explicit BigImageFileDirectory(TIFFFile &aFile);

    ~BigImageFileDirectory() = default;

    BigImageFileDirectory(const BigImageFileDirectory &) = default;
    BigImageFileDirectory(BigImageFileDirectory &&)      = default;

    BigImageFileDirectory &operator=(const BigImageFileDirectory &) = delete;
    BigImageFileDirectory &operator=(BigImageFileDirectory &&)      = delete;

    std::shared_ptr<BigImageFileDirectoryEntry> GetEntry(ushort aTagID);
    std::vector<std::shared_ptr<BigImageFileDirectoryEntry>> &GetEntries()
    {
        return mEntries;
    }
};

class BigIFDImageLength : public BigImageFileDirectoryEntry
{
    uint mValue;

  public:
    static constexpr ushort TagID   = 257;
    static constexpr char TagName[] = "Image length";

    BigIFDImageLength(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] uint Value() const;
};

class BigIFDImageWidth : public BigImageFileDirectoryEntry
{
    uint mValue;

  public:
    static constexpr ushort TagID   = 256;
    static constexpr char TagName[] = "Image width";

    BigIFDImageWidth(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] uint Value() const;
};

class BigIFDRowsPerStrip : public BigImageFileDirectoryEntry
{
    uint mValue;

  public:
    static constexpr ushort TagID   = 278;
    static constexpr char TagName[] = "Rows per strip";

    BigIFDRowsPerStrip(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] uint Value() const;
};

class BigIFDStripByteCounts : public BigImageFileDirectoryEntry
{
    std::vector<ulong64> mValue;

  public:
    static constexpr ushort TagID   = 279;
    static constexpr char TagName[] = "Strip byte counts";

    BigIFDStripByteCounts(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] const std::vector<ulong64> &Value() const;
};

class BigIFDStripOffsets : public BigImageFileDirectoryEntry
{
    std::vector<ulong64> mValue;

  public:
    static constexpr ushort TagID   = 273;
    static constexpr char TagName[] = "Strip offsets";

    BigIFDStripOffsets(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] const std::vector<ulong64> &Value() const;
};

class BigIFDArtist : public BigIFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 315;
    static constexpr char TagName[] = "Artist";

    BigIFDArtist(TIFFFile &aFile, ushort aTagID);
};

class BigIFDCopyright : public BigIFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 33432;
    static constexpr char TagName[] = "Copyright";

    BigIFDCopyright(TIFFFile &aFile, ushort aTagID);
};

class BigIFDDateTime : public BigIFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 306;
    static constexpr char TagName[] = "Date/Time";

    BigIFDDateTime(TIFFFile &aFile, ushort aTagID);
};

class BigIFDHostComputer : public BigIFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 316;
    static constexpr char TagName[] = "Host computer";

    BigIFDHostComputer(TIFFFile &aFile, ushort aTagID);
};

class BigIFDImageDescription : public BigIFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 270;
    static constexpr char TagName[] = "Image description";

    BigIFDImageDescription(TIFFFile &aFile, ushort aTagID);
};

class BigIFDModel : public BigIFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 272;
    static constexpr char TagName[] = "Model";

    BigIFDModel(TIFFFile &aFile, ushort aTagID);
};

class BigIFDMake : public BigIFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 271;
    static constexpr char TagName[] = "Make";

    BigIFDMake(TIFFFile &aFile, ushort aTagID);
};

class BigIFDSoftware : public BigIFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 305;
    static constexpr char TagName[] = "Software";

    BigIFDSoftware(TIFFFile &aFile, ushort aTagID);
};

class BigIFDBitsPerSample : public BigIFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 258;
    static constexpr char TagName[] = "Bits per sample";

    BigIFDBitsPerSample(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] ushort Value(size_t aIdx) const;
};

class BigIFDCellLength : public BigIFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 265;
    static constexpr char TagName[] = "Cell length";

    BigIFDCellLength(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] ushort Value() const;
};

class BigIFDCellWidth : public BigIFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 264;
    static constexpr char TagName[] = "Cell width";

    BigIFDCellWidth(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] ushort Value() const;
};

class BigIFDColorMap : public BigIFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 320;
    static constexpr char TagName[] = "Color map";

    BigIFDColorMap(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] ushort Value() const;
};

class BigIFDCompression : public BigIFDEntry<tiffTag::TIFFCompression>
{
  public:
    static constexpr ushort TagID   = 259;
    static constexpr char TagName[] = "Compression";

    BigIFDCompression(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] tiffTag::TIFFCompression Value() const;
};

class BigIFDExtraSamples : public BigIFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 338;
    static constexpr char TagName[] = "Extra samples";

    BigIFDExtraSamples(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] ushort Value() const;
};

class BigIFDDifferencingPredictor : public BigIFDEntry<tiffTag::TIFFDifferencingPredictor>
{
  public:
    static constexpr ushort TagID   = 317;
    static constexpr char TagName[] = "Differencing Predictor";

    BigIFDDifferencingPredictor(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] tiffTag::TIFFDifferencingPredictor Value() const;
};

class BigIFDFillOrder : public BigIFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 226;
    static constexpr char TagName[] = "Fill order";

    BigIFDFillOrder(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] ushort Value() const;
};

class BigIFDFreeByteCounts : public BigIFDEntry<uint>
{
  public:
    static constexpr ushort TagID   = 289;
    static constexpr char TagName[] = "Free byte counts";

    BigIFDFreeByteCounts(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] uint Value() const;
};

class BigIFDFreeOffsets : public BigIFDEntry<uint>
{
  public:
    static constexpr ushort TagID   = 288;
    static constexpr char TagName[] = "Free offsets";

    BigIFDFreeOffsets(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] uint Value() const;
};

class BigIFDGrayResponseCurve : public BigIFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 291;
    static constexpr char TagName[] = "Gray response curve";

    BigIFDGrayResponseCurve(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] const std::vector<ushort> &Value() const;
};

class BigIFDGrayResponseUnit : public BigIFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 290;
    static constexpr char TagName[] = "Gray response unit";

    BigIFDGrayResponseUnit(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] ushort Value() const;
};

class BigIFDMaxSampleValue : public BigIFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 281;
    static constexpr char TagName[] = "Max sample value";

    BigIFDMaxSampleValue(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] ushort Value() const;
};

class BigIFDMinSampleValue : public BigIFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 280;
    static constexpr char TagName[] = "Min sample value";

    BigIFDMinSampleValue(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] ushort Value() const;
};

class BigIFDNewSubfileType : public BigIFDEntry<uint>
{
  public:
    static constexpr ushort TagID   = 254;
    static constexpr char TagName[] = "New subfile type";

    BigIFDNewSubfileType(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] uint Value() const;
};

class BigIFDOrientation : public BigIFDEntry<tiffTag::TiffOrientation>
{
  public:
    static constexpr ushort TagID   = 274;
    static constexpr char TagName[] = "Orientation";

    BigIFDOrientation(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] tiffTag::TiffOrientation Value() const;
};

class BigIFDPhotometricInterpretation : public BigIFDEntry<tiffTag::TIFFPhotometricInterpretation>
{
  public:
    static constexpr ushort TagID   = 262;
    static constexpr char TagName[] = "Photometric interpretation";

    BigIFDPhotometricInterpretation(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] tiffTag::TIFFPhotometricInterpretation Value() const;
};

class BigIFDPlanarConfiguration : public BigIFDEntry<tiffTag::TIFFPlanarConfigurartion>
{
  public:
    static constexpr ushort TagID   = 284;
    static constexpr char TagName[] = "Planar configuration";

    BigIFDPlanarConfiguration(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] tiffTag::TIFFPlanarConfigurartion Value() const;
};

class BigIFDResolutionUnit : public BigIFDEntry<tiffTag::TIFFResolutionUnit>
{
  public:
    static constexpr ushort TagID   = 296;
    static constexpr char TagName[] = "Resolution unit";

    BigIFDResolutionUnit(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] tiffTag::TIFFResolutionUnit Value() const;
};

class BigIFDSamplesPerPixel : public BigIFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 277;
    static constexpr char TagName[] = "Samples per pixel";

    BigIFDSamplesPerPixel(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] ushort Value() const;
};

class BigIFDSampleFormat : public BigIFDEntry<tiffTag::TIFFSampleFormat>
{
  public:
    static constexpr ushort TagID   = 339;
    static constexpr char TagName[] = "Samples format";

    BigIFDSampleFormat(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] tiffTag::TIFFSampleFormat Value() const;
};

class BigIFDSubfileType : public BigIFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 255;
    static constexpr char TagName[] = "Subfile type";

    BigIFDSubfileType(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] ushort Value() const;
};

class BigIFDThreshholding : public BigIFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 263;
    static constexpr char TagName[] = "Threshholding";

    BigIFDThreshholding(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] ushort Value() const;
};

class BigIFDXResolution : public BigIFDEntry<tiffTag::Rational>
{
  public:
    static constexpr ushort TagID   = 282;
    static constexpr char TagName[] = "X-Resolution";

    BigIFDXResolution(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] tiffTag::Rational Value() const;
};

class BigIFDYResolution : public BigIFDEntry<tiffTag::Rational>
{
  public:
    static constexpr ushort TagID   = 283;
    static constexpr char TagName[] = "Y-Resolution";

    BigIFDYResolution(TIFFFile &aFile, ushort aTagID);

    [[nodiscard]] tiffTag::Rational Value() const;
};

} // namespace opp::fileIO::bigTiffTag