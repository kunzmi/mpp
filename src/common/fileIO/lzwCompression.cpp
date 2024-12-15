#include "lzwCompression.h"
#include <common/defines.h>
#include <common/safeCast.h>
#include <cstring>

namespace opp::fileIO
{
LZWDecoder::LZWDecoder() : mBuffer(CSIZE), dec_codetab(mBuffer.data())
{
    Reset();
}

void LZWDecoder::Reset()
{
    for (uint code = 0; code <= 255; code++) // NOLINT
    {
        dec_codetab[code].value     = byte(code);
        dec_codetab[code].firstchar = byte(code);
        dec_codetab[code].length    = 1;
        dec_codetab[code].next      = nullptr;
    }

    dec_free_entp = dec_codetab + CODE_FIRST; // next free entry
    memset(dec_free_entp, 0, (CSIZE - CODE_FIRST) * sizeof(code_t));
}

bool LZWDecoder::Decode(byte *aData, size_t aSize, void *aDecoded)
{
    byte *op      = reinterpret_cast<byte *>(aDecoded);
    int occ       = to_int(aSize);
    byte *tp      = nullptr;
    byte *bp      = nullptr;
    ushort code   = 0;
    uint len      = 0;
    uint nbits    = BITS_MIN;
    code_t *codep = nullptr;

    bp = aData;
    BitReader bitReader(bp);

    code_t *oldcodep  = nullptr;
    code_t *free_entp = dec_free_entp;
    code_t *maxcodep  = &dec_codetab[MAXCODE(BITS_MIN) - 1];

    // old LZW compression scheme that we don't support
    if (aData[0] == 0 && (aData[1] & 1u) != 0)
    {
        return false;
    }

    while (occ > 0)
    {
        code = bitReader.GetNextCode(nbits);

        if (code == CODE_EOI)
        {
            break;
        }
        if (code == CODE_CLEAR)
        {
            free_entp = dec_codetab + CODE_FIRST;
            nbits     = BITS_MIN;
            maxcodep  = dec_codetab + MAXCODE(BITS_MIN) - 1;

            code = bitReader.GetNextCode(nbits);

            if (code == CODE_EOI)
            {
                break;
            }
            *op = byte(code);
            op++;
            occ--;
            oldcodep = dec_codetab + code;
            continue;
        }
        codep = dec_codetab + code;

        // Add the new entry to the code table.
        if (free_entp < &dec_codetab[0] || free_entp >= &dec_codetab[CSIZE])
        {
            return false;
        }

        free_entp->next = oldcodep;
        if (free_entp->next < &dec_codetab[0] || free_entp->next >= &dec_codetab[CSIZE])
        {
            return false;
        }

        free_entp->firstchar = free_entp->next->firstchar;
        free_entp->length    = to_ushort(free_entp->next->length + 1);
        free_entp->value     = (codep < free_entp) ? codep->firstchar : free_entp->firstchar;
        free_entp++;

        if (free_entp > maxcodep)
        {
            if (++nbits > BITS_MAX) // should not happen
            {
                nbits = BITS_MAX;
            }
            maxcodep = dec_codetab + MAXCODE(nbits) - 1;
        }
        oldcodep = codep;
        if (code >= 256) // NOLINT
        {
            // Code maps to a string, copy string value to output (written in reverse).
            if (codep->length == 0)
            {
                return false;
            }
            if (codep->length > occ)
            {
                return false;
            }
            len = codep->length;
            tp  = op + len;

            do // NOLINT(cppcoreguidelines-avoid-do-while)
            {
                // t;
                --tp;
                const byte t = codep->value;
                codep        = codep->next;
                *tp          = t;
            } while (codep != nullptr && tp > op);

            if (codep != nullptr)
            {
                return false;
            }
            op += len;
            occ -= to_int(len);
        }
        else
        {
            *op = byte(code);
            op++;
            occ--;
        }
    }

    return occ == 0;
}

BitReader::BitReader(byte *&aDataStream) : mDataStream(aDataStream)
{
}

ushort BitReader::GetNextCode(uint aNumberOfBits)
{
    uint value = *mDataStream;
    mDataStream++;

    mNextdata = (mNextdata << 8u) | value;

    mNextbits += 8;
    if (mNextbits < aNumberOfBits)
    {
        value = *mDataStream;
        mDataStream++;
        mNextdata = (mNextdata << 8u) | value;
        mNextbits += 8;
    }
    const uint bitmask = LZWDecoder::MAXCODE(aNumberOfBits);
    const ushort code  = to_ushort((mNextdata >> (mNextbits - aNumberOfBits)) & bitmask);
    mNextbits -= aNumberOfBits;
    return code;
}
} // namespace opp::fileIO