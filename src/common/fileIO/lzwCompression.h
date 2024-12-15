#pragma once
#include <common/defines.h>
#include <vector>

namespace opp::fileIO
{
// code is taken to large extent from libTiff and its lzw extension (tif_lzw.c)

// Decoding-specific state.
struct code_t
{
    code_t *next;
    ushort length;  // string len, including this token
    byte value;     // data value
    byte firstchar; // first token of string
};

/// <summary>
/// LZW decoder utlity class
/// </summary>
class LZWDecoder
{
  public:
    LZWDecoder();
    void Reset();
    bool Decode(byte *aData, size_t aSizeDecoded, void *aDecoded);

    static constexpr uint MAXCODE(uint aN)
    {
        return (1U << aN) - 1U;
    }

  private:
    static constexpr uint BITS_MIN = 9;  // start with 9 bits
    static constexpr uint BITS_MAX = 12; // max of 12 bit strings
    // predefined codes
    static constexpr ushort CODE_CLEAR = 256; // code to clear string table
    static constexpr ushort CODE_EOI   = 257; // end-of-information code
    static constexpr ushort CODE_FIRST = 258; // first free code entry
    static constexpr uint CSIZE        = (1U << BITS_MAX) - 1U + 1U;

    std::vector<code_t> mBuffer;
    code_t *dec_codetab;

    code_t *dec_free_entp{nullptr}; // next free entry
};

/// <summary>
/// Reads bitwise from an data array
/// </summary>
class BitReader
{
  public:
    explicit BitReader(byte *&aDataStream);

    ~BitReader() = default;

    BitReader(const BitReader &) = delete;
    BitReader(BitReader &&)      = delete;

    BitReader &operator=(const BitReader &) = delete;
    BitReader &operator=(BitReader &&)      = delete;

    ushort GetNextCode(uint aNumberOfBits);

  private:
    byte *&mDataStream; // Rererence to the original array
    uint mNextdata{};
    uint mNextbits{};
};
} // namespace opp::fileIO