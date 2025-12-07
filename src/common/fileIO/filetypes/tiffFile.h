#pragma once

#include "../dllexport_fileio.h"
#include "bigTiffImageFileDirectory.h"
#include "tiffImageFileDirectory.h"
#include <common/defines.h>
#include <common/fileIO/fileReader.h>
#include <common/fileIO/fileWriter.h>
#include <common/fileIO/lzwCompression.h>
#include <common/vectorTypes.h>
#include <filesystem>
#include <memory>
#include <vector>

namespace mpp::fileIO
{
struct MPPEXPORT_COMMON_FILEIO TiffFileHeader
{
    ushort BytaOrder;
    ushort ID;
    uint OffsetToIFD;
};

struct MPPEXPORT_COMMON_FILEIO BigTiffFileHeader
{
    ushort BytaOrder;
    ushort ID;
    ushort OffsetByteSize;
    ushort MustBeZero;
    ulong64 OffsetToIFD;
};

struct MPPEXPORT_COMMON_FILEIO TiffReadElement
{
    size_t DataSize;
    size_t DataOffset;
    size_t DestinationOffset;
    size_t RowsPerStrip;
};

/// <summary>
/// TIFFFile represents a *.tif file, a common image file format
/// </summary>
class MPPEXPORT_COMMON_FILEIO TIFFFile : public FileReader, public FileWriter
{
  private:
    TiffFileHeader mTiffFileHeader{};
    BigTiffFileHeader mBigTiffFileHeader{};

    std::vector<char> mData;
    bool mIsBigTiff{false};
    std::vector<std::shared_ptr<tiffTag::ImageFileDirectory>> mImageFileDirectories;
    std::vector<std::shared_ptr<bigTiffTag::BigImageFileDirectory>> mBigImageFileDirectories;
    int mWidth{0};
    int mHeight{0};
    int mPlanes{0};
    int mBitsPerSample{0};
    int mSamplesPerPixel{0};
    double mPixelSize{0};
    tiffTag::TiffOrientation mOrientation{tiffTag::TiffOrientation::TOPLEFT};
    bool mIsPlanar{false};
    tiffTag::TIFFSampleFormat mSampleFormat{tiffTag::TIFFSampleFormat::VOIDTYPE};
    tiffTag::TIFFCompression mCompression{tiffTag::TIFFCompression::NoCompression};
    tiffTag::TIFFPhotometricInterpretation mPhotometricInterpretation{
        tiffTag::TIFFPhotometricInterpretation::BlackIsZero};
    tiffTag::TIFFDifferencingPredictor mDifferencingPredictor{tiffTag::TIFFDifferencingPredictor::None};

    std::vector<std::vector<TiffReadElement>> mReadSegments;

    [[nodiscard]] mpp::image::PixelTypeEnum GetDataTypeUnsigned() const;
    [[nodiscard]] mpp::image::PixelTypeEnum GetDataTypeSigned() const;
    [[nodiscard]] mpp::image::PixelTypeEnum GetDataTypeFloat() const;
    [[nodiscard]] mpp::image::PixelTypeEnum GetDataTypeComplex() const;
    [[nodiscard]] mpp::image::PixelTypeEnum GetDataTypeComplexFloat() const;

    void ReadTIFF();
    void ReadBigTIFF();
    void ReadPlane(size_t aIdx);
    void ReadPlaneNoCompression(size_t aIdx);
    void ReadPlaneLZWCompression(size_t aIdx);
    void ReadPlaneDeflateCompression(size_t aIdx);
    void ReadPlanePackBitsCompression(size_t aIdx);
    void DecodeDifferencingPredictor(size_t aIdx);

    static void EncodeDifferencingPredictor(void *aData, mpp::image::PixelTypeEnum aDataType, uint aWidth,
                                            uint aHeight);

  public:
    /// <summary>
    /// Creates a new TIFFFile instance. The file name is only set internally; the file itself keeps untouched.
    /// </summary>
    explicit TIFFFile(const std::filesystem::path &aFileName);

    /// <summary>
    /// Creates a new TIFFFile instance in memory that can be saved to disk later.
    /// </summary>
    TIFFFile(const mpp::image::Size2D &aSize, mpp::image::PixelTypeEnum aDataType, double aPixelSize = 1.0);

    ~TIFFFile() override = default;

    TIFFFile(const TIFFFile &) = default;
    TIFFFile(TIFFFile &&)      = default;

    TIFFFile &operator=(const TIFFFile &) = default;
    TIFFFile &operator=(TIFFFile &&)      = default;

    // overrides:

    /// <summary>
    /// Converts from Tiff data type enum to internal data type
    /// </summary>
    [[nodiscard]] mpp::image::PixelTypeEnum GetDataType() const override;

    /// <summary>
    /// Converts from internal data type to Tiff data type enum<para/>
    /// </summary>
    void SetDataType(mpp::image::PixelTypeEnum aDataType) override;

    /// <summary>
    /// Opens the file and reads the entire content.
    /// </summary>
    void OpenAndRead() override;

    /// <summary>
    /// Opens the file and reads only the file header.
    /// </summary>
    void OpenAndReadHeader() override;

    /// <summary>
    /// Opens the file and reads only the file header (no exception: returns false on failure).
    /// </summary>
    [[nodiscard]] bool TryToOpenAndReadHeader() noexcept override;

    /// <summary>
    /// Opens the file and writes the entire content.
    /// </summary>
    void OpenAndWrite() override;

    /// <summary>
    /// Saves the content of the in-memory data to file aFileName and sets the internal filename
    /// </summary>
    void SaveAs(const std::filesystem::path &aFileName) override;

    /// <summary>
    /// Returns the size of the data block. If the header is not yet read, it will return 0.
    /// </summary>
    [[nodiscard]] size_t DataSize() const override;

    /// <summary>
    /// Returns the size of one 2D image slice in bytes. If the header is not yet read, it will return 0.
    /// For planar images, GetImageSizeInBytes() returns only the size for one color channel.
    /// </summary>
    [[nodiscard]] size_t GetImageSizeInBytes() const override;

    /// <summary>
    /// Returns the inner data pointer.
    /// </summary>
    [[nodiscard]] void *Data() override;

    /// <summary>
    /// Returns the inner data pointer shifted to image plane aIdx (IFD index for TIFF).
    /// </summary>
    [[nodiscard]] void *Data(size_t aIdx) override;

    /// <summary>
    /// Returns the inner data pointer shifted to image plane aIdx (IFD index for TIFF) amd color channel aColorChannel
    /// (for planar images).
    /// </summary>
    [[nodiscard]] void *Data(size_t aIdx, size_t aColorChannel);

    /// <summary>
    /// Returns the image dimensions stored in the file header.
    /// </summary>
    [[nodiscard]] Vec3i Size() const override;

    /// <summary>
    /// Returns the dimensions of an X/Y-image plane stored in the file header.
    /// </summary>
    [[nodiscard]] mpp::image::Size2D SizePlane() const override;

    /// <summary>
    /// Returns the pixel size stored in file header converted to nm.
    /// </summary>
    [[nodiscard]] double PixelSize() const override;

    /// <summary>
    /// Sets the pixel size given in nm and converts them if needed to the internal unit.
    /// </summary>
    void SetPixelSize(double aPixelSizeInNM) override;

    /// <summary>
    /// Reads a specific slice of a 3D data set (movie stack, tilt-series, volume) from file.
    /// </summary>
    void ReadSlice(size_t aIdx) override;

    /// <summary>
    /// Reads specific slices of a 3D data set (movie stack, tilt-series, volume) from file.
    /// </summary>
    void ReadSlices(size_t aStartIdx, size_t aSliceCount) override;

    /// <summary>
    /// Reads a specific slice of a 3D data set (movie stack, tilt-series, volume) from file.<para/>
    /// This only copies data from the internal storage buffer
    /// </summary>
    void ReadSlice(void *aData, size_t aIdx) override;

    /// <summary>
    /// Reads specific slices of a 3D data set (movie stack, tilt-series, volume) from file.<para/>
    /// This only copies data from the internal storage buffer
    /// </summary>
    void ReadSlices(void *aData, size_t aStartIdx, size_t aSliceCount) override;

    /// <summary>
    /// Reads raw data from file to user provided memory. Offset is from the start of data block without any
    /// header.<para/> This only copies data from the internal storage buffer
    /// </summary>
    void ReadRaw(void *aData, size_t aSizeInBytes, size_t aOffset = 0) override;

    /// <summary>
    /// Returns the file type: TIFF
    /// </summary>
    [[nodiscard]] FileType GetFileType() const override;

    // Tiff file specific:

    /// <summary>
    /// Returns the image width
    /// </summary>
    [[nodiscard]] int Width() const;

    /// <summary>
    /// Returns the image height
    /// </summary>
    [[nodiscard]] int Height() const;

    /// <summary>
    /// Returns the image depth / number of planes
    /// </summary>
    [[nodiscard]] int Depth() const;

    /// <summary>
    /// Returns the number of bits per sample
    /// </summary>
    [[nodiscard]] int BitsPerSample() const;

    /// <summary>
    /// Returns the number of samples per pixel
    /// </summary>
    [[nodiscard]] int SamplesPerPixel() const;

    /// <summary>
    /// Returns true if image is stored in planar layout and not interleaved
    /// </summary>
    [[nodiscard]] bool IsPlanar() const;

    /// <summary>
    /// Returns the orientation information stored in TIFF file
    /// </summary>
    [[nodiscard]] tiffTag::TiffOrientation Orientation() const;

    /// <summary>
    /// Determines if a given image dimension and datatype can be written to a Tiff file
    /// </summary>
    [[nodiscard]] static bool CanWriteAs(int aDimX, int aDimY, mpp::image::PixelTypeEnum aDatatype);

    /// <summary>
    /// Determines if a given image dimension and datatype can be written to a Tiff file
    /// </summary>
    [[nodiscard]] static bool CanWriteAs(const Vec2i &aDim, mpp::image::PixelTypeEnum aDatatype);

    /// <summary>
    /// Determines if a given image dimension and datatype can be written to a Tiff file
    /// </summary>
    [[nodiscard]] static bool CanWriteAs(const Vec3i &aDim, mpp::image::PixelTypeEnum aDatatype);

    /// <summary>
    /// Writes data to a tiff file
    /// </summary>
    static void WriteTIFF(const std::filesystem::path &aFileName, int aDimX, int aDimY, double aPixelSize,
                          mpp::image::PixelTypeEnum aDatatype, const void *aData);

    /// <summary>
    /// Writes data to a tiff file with compression (input data will be modified!)
    /// </summary>
    static void WriteTIFF(const std::filesystem::path &aFileName, int aDimX, int aDimY, double aPixelSize,
                          mpp::image::PixelTypeEnum aDatatype, void *aData, int aZIPCompressionLevel);

    /// <summary>
    /// Writes data to a tiff file. Pass nullptr if less than 4 color planes. (planar color planes)
    /// </summary>
    static void WriteTIFF(const std::filesystem::path &aFileName, int aDimX, int aDimY, double aPixelSize,
                          mpp::image::PixelTypeEnum aDatatype, const void *aData0, const void *aData1,
                          const void *aData2, const void *aData3);

    /// <summary>
    /// Writes data to a tiff file with compression. Pass nullptr if less than 4 color planes. (planar color planes,
    /// input data will be modified!)
    /// </summary>
    static void WriteTIFF(const std::filesystem::path &aFileName, int aDimX, int aDimY, double aPixelSize,
                          mpp::image::PixelTypeEnum aDatatype, void *aData0, void *aData1, void *aData2, void *aData3,
                          int aZIPCompressionLevel);

    [[nodiscard]] bool IsLittleEndian();

    friend class tiffTag::ImageFileDirectory;
    friend class tiffTag::ImageFileDirectoryEntry;
    friend class bigTiffTag::BigImageFileDirectory;
    friend class bigTiffTag::BigImageFileDirectoryEntry;
};
} // namespace mpp::fileIO