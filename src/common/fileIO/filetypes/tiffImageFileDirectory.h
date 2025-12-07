#pragma once
#include "../dllexport_fileio.h"
#include <common/defines.h>
#include <common/fileIO/pseudoFileReader.h>
#include <filesystem>
#include <memory>
#include <vector>

namespace mpp::fileIO
{
// forward declaration
class TIFFFile;

namespace tiffTag
{

class MPPEXPORT_COMMON_FILEIO SaveTiffTag
{
  protected:
  public:
    SaveTiffTag()          = default;
    virtual ~SaveTiffTag() = default;

    SaveTiffTag(const SaveTiffTag &) = default;
    SaveTiffTag(SaveTiffTag &&)      = default;

    SaveTiffTag &operator=(const SaveTiffTag &) = default;
    SaveTiffTag &operator=(SaveTiffTag &&)      = default;

    virtual void SavePass1(std::ostream &aStream) = 0;

    virtual void SavePass2(std::ostream &aStream) = 0;
};

class MPPEXPORT_COMMON_FILEIO Rational
{
  public:
    uint nominator{0};   // NOLINT(misc-non-private-member-variables-in-classes)
    uint denominator{1}; // NOLINT(misc-non-private-member-variables-in-classes)

    Rational(uint aNominator, uint aDenominator);
    explicit Rational(uint aValues[2]);
    Rational() = default;

    ~Rational() = default;

    Rational(const Rational &) = default;
    Rational(Rational &&)      = default;

    Rational &operator=(const Rational &) = default;
    Rational &operator=(Rational &&)      = default;

    [[nodiscard]] double GetValue() const;
};

class MPPEXPORT_COMMON_FILEIO SRational
{
  public:
    int nominator{0};   // NOLINT(misc-non-private-member-variables-in-classes)
    int denominator{1}; // NOLINT(misc-non-private-member-variables-in-classes)

    SRational(int aNominator, int aDenominator);
    explicit SRational(int aValues[2]);
    SRational() = default;

    ~SRational() = default;

    SRational(const SRational &) = default;
    SRational(SRational &&)      = default;

    SRational &operator=(const SRational &) = default;
    SRational &operator=(SRational &&)      = default;

    [[nodiscard]] double GetValue() const;
};

Rational TIFFConvertPixelSizeToDPI(double aPixelSize);

double TIFFConvertDPIToPixelSize(const Rational &aDPI);

enum class TiffType : ushort // NOLINT(performance-enum-size)
{
    BYTE      = 1,
    ASCII     = 2,
    SHORT     = 3,
    LONG      = 4,
    RATIONAL  = 5,
    SBYTE     = 6,
    UNDEFINED = 7,
    SSHORT    = 8,
    SLONG     = 9,
    SRATIONAL = 10,
    FLOAT     = 11,
    DOUBLE    = 12,
    LONG8     = 16,
    SLONG8    = 17,
    IFD8      = 18
};

size_t GetTiffTypeSizeInBytes(TiffType aType);

struct MPPEXPORT_COMMON_FILEIO TiffTag
{
    ushort TagID;
    TiffType Type;
    uint Count;
    union
    {
        char CharVal;
        byte UCharVal;
        short ShortVal;
        ushort UShortVal;
        int IntVal;
        uint UIntVal;
    } Offset;
};

enum class TIFFSampleFormat : ushort // NOLINT(performance-enum-size)
{
    UINT          = 1,
    INT           = 2,
    IEEEFP        = 3,
    VOIDTYPE      = 4,
    COMPLEXINT    = 5,
    COMPLEXIEEEFP = 6
};

enum class TIFFCompression : ushort
{
    NoCompression = 1,
    CCITTGroup3   = 2,
    LZW           = 5,
    DeflateAdobe  = 8,
    PackBits      = 32773,
    Deflate       = 32946,
    EER8Bit       = 65000,
    EER7Bit       = 65001
};

enum class TiffOrientation : ushort // NOLINT(performance-enum-size)
{
    TOPLEFT  = 1,
    TOPRIGHT = 2,
    BOTRIGHT = 3,
    BOTLEFT  = 4,
    LEFTTOP  = 5,
    RIGHTTOP = 6,
    RIGHTBOT = 7,
    LEFTBOT  = 8
};

enum class TIFFPhotometricInterpretation : ushort // NOLINT(performance-enum-size)
{
    WhiteIsZero      = 0,
    BlackIsZero      = 1,
    RGB              = 2,
    Palette          = 3,
    TransparencyMask = 4,
    CMYK             = 5,
    YCbCr            = 6,
    CIELab           = 8
};

enum class TIFFPlanarConfigurartion : ushort // NOLINT(performance-enum-size)
{
    Chunky = 1,
    Planar = 2
};

enum class TIFFResolutionUnit : ushort // NOLINT(performance-enum-size)
{
    None       = 1,
    Inch       = 2,
    Centimeter = 3
};

enum class TIFFDifferencingPredictor : ushort // NOLINT(performance-enum-size)
{
    None                   = 1,
    HorizontalDifferencing = 2
};

class MPPEXPORT_COMMON_FILEIO ImageFileDirectoryEntry : public PseudoFileReader
{
  protected:
    TiffTag mTag;           // NOLINT -> should be private
    size_t mOffsetInStream; // NOLINT -> should be private

    ImageFileDirectoryEntry(TIFFFile &aFile, ushort aTagID);

    ImageFileDirectoryEntry(ushort aTagID, TiffType aFieldType, uint aValueCount);

    virtual size_t WriteEntryHeader(uint aOffsetOrValue, std::ostream &aStream, int aValueCount = -1);

    virtual void WritePass2(const char *aData, size_t aDataLength, std::ostream &aStream);

  public:
    explicit ImageFileDirectoryEntry(TIFFFile &aFile);
    static std::shared_ptr<ImageFileDirectoryEntry> CreateFileDirectoryEntry(TIFFFile &aFile);

    virtual void SavePass1(std::ostream &aStream);

    virtual void SavePass2(std::ostream &aStream);

    friend class ImageFileDirectory;

    [[nodiscard]] ushort GetTagID() const
    {
        return mTag.TagID;
    }
};

template <typename T> class MPPEXPORT_COMMON_FILEIO IFDEntry : public ImageFileDirectoryEntry
{
  protected:
    std::vector<T> mValue; // NOLINT -> should be private

  public:
    IFDEntry(TIFFFile &aFile, ushort aTagID);

    IFDEntry(T aValue, ushort aTagID, TiffType aFieldType);

    IFDEntry(std::vector<T> &&aValues, ushort aTagID, TiffType aFieldType);

    void SavePass1(std::ostream &aStream) override;

    void SavePass2(std::ostream &aStream) override;
};

template <> class IFDEntry<std::string> : public ImageFileDirectoryEntry
{
  protected:
    std::string mValue; // NOLINT -> should be private

  public:
    IFDEntry(TIFFFile &aFile, ushort aTagID);

    IFDEntry(const std::string &aValue, ushort aTagID);

    [[nodiscard]] const std::string &Value() const;

    void SavePass1(std::ostream &aStream) override;

    void SavePass2(std::ostream &aStream) override;
};

class MPPEXPORT_COMMON_FILEIO ImageFileDirectory
{
  private:
    ushort mEntryCount{0};
    std::vector<std::shared_ptr<ImageFileDirectoryEntry>> mEntries;

  public:
    explicit ImageFileDirectory(TIFFFile &aFile);

    ImageFileDirectory(uint aWidth, uint aHeight, double aPixelSize, ushort aBitPerSample, ushort aSamplesPerPixel,
                       TIFFSampleFormat aSampleFormat, bool aPlanar,
                       TIFFPhotometricInterpretation aPhotometricInterpretation);

    ImageFileDirectory(uint aWidth, uint aHeight, double aPixelSize, ushort aBitPerSample, ushort aSamplesPerPixel,
                       TIFFSampleFormat aSampleFormat, bool aPlanar,
                       TIFFPhotometricInterpretation aPhotometricInterpretation, bool aDifference,
                       std::vector<uint> aCompressedSize);

    ~ImageFileDirectory() = default;

    ImageFileDirectory(const ImageFileDirectory &) = default;
    ImageFileDirectory(ImageFileDirectory &&)      = default;

    ImageFileDirectory &operator=(const ImageFileDirectory &) = default;
    ImageFileDirectory &operator=(ImageFileDirectory &&)      = default;

    std::shared_ptr<ImageFileDirectoryEntry> GetEntry(ushort aTagID);
    std::vector<std::shared_ptr<ImageFileDirectoryEntry>> &GetEntries()
    {
        return mEntries;
    }

    void SaveAsTiff(std::ofstream &aStream, void *aData, size_t aDataSize);
};

class MPPEXPORT_COMMON_FILEIO IFDImageLength : public ImageFileDirectoryEntry
{
    uint mValue;

  public:
    static constexpr ushort TagID   = 257;
    static constexpr char TagName[] = "Image length";

    IFDImageLength(TIFFFile &aFile, ushort aTagID);

    explicit IFDImageLength(uint aValue);

    [[nodiscard]] uint Value() const;

    void SavePass1(std::ostream &aStream) override;
};

class MPPEXPORT_COMMON_FILEIO IFDImageWidth : public ImageFileDirectoryEntry
{
    uint mValue;

  public:
    static constexpr ushort TagID   = 256;
    static constexpr char TagName[] = "Image width";

    IFDImageWidth(TIFFFile &aFile, ushort aTagID);

    explicit IFDImageWidth(uint aValue);

    [[nodiscard]] uint Value() const;

    void SavePass1(std::ostream &aStream) override;
};

class MPPEXPORT_COMMON_FILEIO IFDRowsPerStrip : public ImageFileDirectoryEntry
{
    uint mValue;

  public:
    static constexpr ushort TagID   = 278;
    static constexpr char TagName[] = "Rows per strip";

    IFDRowsPerStrip(TIFFFile &aFile, ushort aTagID);

    explicit IFDRowsPerStrip(uint aValue);

    [[nodiscard]] uint Value() const;

    void SavePass1(std::ostream &aStream) override;
};

class MPPEXPORT_COMMON_FILEIO IFDStripByteCounts : public ImageFileDirectoryEntry
{
    std::vector<uint> mValue;

  public:
    static constexpr ushort TagID   = 279;
    static constexpr char TagName[] = "Strip byte counts";

    IFDStripByteCounts(TIFFFile &aFile, ushort aTagID);

    explicit IFDStripByteCounts(std::vector<uint> &&aValue);

    explicit IFDStripByteCounts(uint aValue);

    [[nodiscard]] const std::vector<uint> &Value() const;

    void SavePass1(std::ostream &aStream) override;

    void SavePass2(std::ostream &aStream) override;
};

class MPPEXPORT_COMMON_FILEIO IFDStripOffsets : public ImageFileDirectoryEntry
{
    std::vector<uint> mValue;

  public:
    static constexpr ushort TagID   = 273;
    static constexpr char TagName[] = "Strip offsets";

    IFDStripOffsets(TIFFFile &aFile, ushort aTagID);

    explicit IFDStripOffsets(size_t aStripCount);

    [[nodiscard]] const std::vector<uint> &Value() const;

    void SavePass1(std::ostream &aStream) override;

    void SavePass2(std::ostream &aStream) override;

    void SaveFinalOffsets(std::ostream &aStream, std::vector<uint> &aFinalOffsets);
};

class MPPEXPORT_COMMON_FILEIO IFDArtist : public IFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 315;
    static constexpr char TagName[] = "Artist";

    IFDArtist(TIFFFile &aFile, ushort aTagID);

    explicit IFDArtist(const std::string &aValue);
};

class MPPEXPORT_COMMON_FILEIO IFDCopyright : public IFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 33432;
    static constexpr char TagName[] = "Copyright";

    IFDCopyright(TIFFFile &aFile, ushort aTagID);

    explicit IFDCopyright(const std::string &aValue);
};

class MPPEXPORT_COMMON_FILEIO IFDDateTime : public IFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 306;
    static constexpr char TagName[] = "Date/Time";

    IFDDateTime(TIFFFile &aFile, ushort aTagID);

    explicit IFDDateTime(const std::string &aValue);
};

class MPPEXPORT_COMMON_FILEIO IFDHostComputer : public IFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 316;
    static constexpr char TagName[] = "Host computer";

    IFDHostComputer(TIFFFile &aFile, ushort aTagID);

    explicit IFDHostComputer(const std::string &aValue);
};

class MPPEXPORT_COMMON_FILEIO IFDImageDescription : public IFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 270;
    static constexpr char TagName[] = "Image description";

    IFDImageDescription(TIFFFile &aFile, ushort aTagID);

    explicit IFDImageDescription(const std::string &aValue);
};

class MPPEXPORT_COMMON_FILEIO IFDModel : public IFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 272;
    static constexpr char TagName[] = "Model";

    IFDModel(TIFFFile &aFile, ushort aTagID);

    explicit IFDModel(const std::string &aValue);
};

class MPPEXPORT_COMMON_FILEIO IFDMake : public IFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 271;
    static constexpr char TagName[] = "Make";

    IFDMake(TIFFFile &aFile, ushort aTagID);

    explicit IFDMake(const std::string &aValue);
};

class MPPEXPORT_COMMON_FILEIO IFDSoftware : public IFDEntry<std::string>
{
  public:
    static constexpr ushort TagID   = 305;
    static constexpr char TagName[] = "Software";

    IFDSoftware(TIFFFile &aFile, ushort aTagID);

    explicit IFDSoftware(const std::string &aValue);
};

class MPPEXPORT_COMMON_FILEIO IFDBitsPerSample : public IFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 258;
    static constexpr char TagName[] = "Bits per sample";

    IFDBitsPerSample(TIFFFile &aFile, ushort aTagID);

    explicit IFDBitsPerSample(ushort aValue);

    explicit IFDBitsPerSample(std::vector<ushort> &aValue);

    [[nodiscard]] ushort Value(size_t aIdx) const;
};

class MPPEXPORT_COMMON_FILEIO IFDCellLength : public IFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 265;
    static constexpr char TagName[] = "Cell length";

    IFDCellLength(TIFFFile &aFile, ushort aTagID);

    explicit IFDCellLength(ushort aValue);

    [[nodiscard]] ushort Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDCellWidth : public IFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 264;
    static constexpr char TagName[] = "Cell width";

    IFDCellWidth(TIFFFile &aFile, ushort aTagID);

    explicit IFDCellWidth(ushort aValue);

    [[nodiscard]] ushort Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDColorMap : public IFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 320;
    static constexpr char TagName[] = "Color map";

    IFDColorMap(TIFFFile &aFile, ushort aTagID);

    explicit IFDColorMap(std::vector<ushort> &aValues);

    [[nodiscard]] const std::vector<ushort> &Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDCompression : public IFDEntry<TIFFCompression>
{
  public:
    static constexpr ushort TagID   = 259;
    static constexpr char TagName[] = "Compression";

    IFDCompression(TIFFFile &aFile, ushort aTagID);

    explicit IFDCompression(TIFFCompression aValue);

    [[nodiscard]] TIFFCompression Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDExtraSamples : public IFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 338;
    static constexpr char TagName[] = "Extra samples";

    IFDExtraSamples(TIFFFile &aFile, ushort aTagID);

    explicit IFDExtraSamples(ushort aValue);

    [[nodiscard]] ushort Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDDifferencingPredictor : public IFDEntry<TIFFDifferencingPredictor>
{
  public:
    static constexpr ushort TagID   = 317;
    static constexpr char TagName[] = "Differencing Predictor";

    IFDDifferencingPredictor(TIFFFile &aFile, ushort aTagID);

    explicit IFDDifferencingPredictor(TIFFDifferencingPredictor aValue);

    [[nodiscard]] TIFFDifferencingPredictor Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDFillOrder : public IFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 226;
    static constexpr char TagName[] = "Fill order";

    IFDFillOrder(TIFFFile &aFile, ushort aTagID);

    explicit IFDFillOrder(ushort aValue);

    [[nodiscard]] ushort Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDFreeByteCounts : public IFDEntry<uint>
{
  public:
    static constexpr ushort TagID   = 289;
    static constexpr char TagName[] = "Free byte counts";

    IFDFreeByteCounts(TIFFFile &aFile, ushort aTagID);

    explicit IFDFreeByteCounts(uint aValue);

    [[nodiscard]] uint Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDFreeOffsets : public IFDEntry<uint>
{
  public:
    static constexpr ushort TagID   = 288;
    static constexpr char TagName[] = "Free offsets";

    IFDFreeOffsets(TIFFFile &aFile, ushort aTagID);

    explicit IFDFreeOffsets(uint aValue);

    [[nodiscard]] uint Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDGrayResponseCurve : public IFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 291;
    static constexpr char TagName[] = "Gray response curve";

    IFDGrayResponseCurve(TIFFFile &aFile, ushort aTagID);

    explicit IFDGrayResponseCurve(std::vector<ushort> &aValues);

    [[nodiscard]] const std::vector<ushort> &Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDGrayResponseUnit : public IFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 290;
    static constexpr char TagName[] = "Gray response unit";

    IFDGrayResponseUnit(TIFFFile &aFile, ushort aTagID);

    explicit IFDGrayResponseUnit(ushort aValue);

    [[nodiscard]] ushort Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDMaxSampleValue : public IFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 281;
    static constexpr char TagName[] = "Max sample value";

    IFDMaxSampleValue(TIFFFile &aFile, ushort aTagID);

    explicit IFDMaxSampleValue(ushort aValue);

    [[nodiscard]] ushort Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDMinSampleValue : public IFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 280;
    static constexpr char TagName[] = "Min sample value";

    IFDMinSampleValue(TIFFFile &aFile, ushort aTagID);

    explicit IFDMinSampleValue(ushort aValue);

    [[nodiscard]] ushort Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDNewSubfileType : public IFDEntry<uint>
{
  public:
    static constexpr ushort TagID   = 254;
    static constexpr char TagName[] = "New subfile type";

    IFDNewSubfileType(TIFFFile &aFile, ushort aTagID);

    explicit IFDNewSubfileType(uint aValue);

    [[nodiscard]] uint Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDOrientation : public IFDEntry<TiffOrientation>
{
  public:
    static constexpr ushort TagID   = 274;
    static constexpr char TagName[] = "Orientation";

    IFDOrientation(TIFFFile &aFile, ushort aTagID);

    explicit IFDOrientation(TiffOrientation aValue);

    [[nodiscard]] TiffOrientation Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDPhotometricInterpretation : public IFDEntry<TIFFPhotometricInterpretation>
{
  public:
    static constexpr ushort TagID   = 262;
    static constexpr char TagName[] = "Photometric interpretation";

    IFDPhotometricInterpretation(TIFFFile &aFile, ushort aTagID);

    explicit IFDPhotometricInterpretation(TIFFPhotometricInterpretation aValue);

    [[nodiscard]] TIFFPhotometricInterpretation Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDPlanarConfiguration : public IFDEntry<TIFFPlanarConfigurartion>
{
  public:
    static constexpr ushort TagID   = 284;
    static constexpr char TagName[] = "Planar configuration";

    IFDPlanarConfiguration(TIFFFile &aFile, ushort aTagID);

    explicit IFDPlanarConfiguration(TIFFPlanarConfigurartion aValue);

    [[nodiscard]] TIFFPlanarConfigurartion Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDResolutionUnit : public IFDEntry<TIFFResolutionUnit>
{
  public:
    static constexpr ushort TagID   = 296;
    static constexpr char TagName[] = "Resolution unit";

    IFDResolutionUnit(TIFFFile &aFile, ushort aTagID);

    explicit IFDResolutionUnit(TIFFResolutionUnit aValue);

    [[nodiscard]] TIFFResolutionUnit Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDSamplesPerPixel : public IFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 277;
    static constexpr char TagName[] = "Samples per pixel";

    IFDSamplesPerPixel(TIFFFile &aFile, ushort aTagID);

    explicit IFDSamplesPerPixel(ushort aValue);

    [[nodiscard]] ushort Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDSampleFormat : public IFDEntry<TIFFSampleFormat>
{
  public:
    static constexpr ushort TagID   = 339;
    static constexpr char TagName[] = "Samples format";

    IFDSampleFormat(TIFFFile &aFile, ushort aTagID);

    explicit IFDSampleFormat(TIFFSampleFormat aValue);

    [[nodiscard]] TIFFSampleFormat Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDSubfileType : public IFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 255;
    static constexpr char TagName[] = "Subfile type";

    IFDSubfileType(TIFFFile &aFile, ushort aTagID);

    explicit IFDSubfileType(ushort aValue);

    [[nodiscard]] ushort Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDThreshholding : public IFDEntry<ushort>
{
  public:
    static constexpr ushort TagID   = 263;
    static constexpr char TagName[] = "Threshholding";

    IFDThreshholding(TIFFFile &aFile, ushort aTagID);

    explicit IFDThreshholding(ushort aValue);

    [[nodiscard]] ushort Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDXResolution : public IFDEntry<Rational>
{
  public:
    static constexpr ushort TagID   = 282;
    static constexpr char TagName[] = "X-Resolution";

    IFDXResolution(TIFFFile &aFile, ushort aTagID);

    explicit IFDXResolution(Rational aValue);

    [[nodiscard]] Rational Value() const;
};

class MPPEXPORT_COMMON_FILEIO IFDYResolution : public IFDEntry<Rational>
{
  public:
    static constexpr ushort TagID   = 283;
    static constexpr char TagName[] = "Y-Resolution";

    IFDYResolution(TIFFFile &aFile, ushort aTagID);

    explicit IFDYResolution(Rational aValue);

    [[nodiscard]] Rational Value() const;
};

} // namespace tiffTag
} // namespace mpp::fileIO