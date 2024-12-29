#include <cfloat>
#include <cmath>
#include <common/bfloat16.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/vector1.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace opp
{
namespace cuda
{

__global__ void test_vector1_kernel(Vector1<byte> *aDataByte, Vector1<byte> *aCompByte,         //
                                    Vector1<sbyte> *aDataSByte, Vector1<byte> *aCompSbyte,      //
                                    Vector1<short> *aDataShort, Vector1<byte> *aCompShort,      //
                                    Vector1<ushort> *aDataUShort, Vector1<byte> *aCompUShort,   //
                                    Vector1<BFloat16> *aDataBFloat, Vector1<byte> *aCompBFloat, //
                                    Vector1<HalfFp16> *aDataHalf, Vector1<byte> *aCompHalf,     //
                                    Vector1<float> *aDataFloat, Vector1<byte> *aCompFloat)
{
    if (threadIdx.x > 0 || blockIdx.x > 6)
    {
        return;
    }

    // sbyte
    if (blockIdx.x == 0)
    {
        Vector1<sbyte> sbA(12);
        Vector1<sbyte> sbB(120);
        Vector1<sbyte> sbC(12);

        aDataSByte[0] = sbA + sbB;
        aDataSByte[1] = sbA - sbB;
        aDataSByte[2] = sbA * sbB;
        aDataSByte[3] = sbA / sbB;

        aDataSByte[4] = sbA;
        aDataSByte[4] += sbB;
        aDataSByte[5] = sbA;
        aDataSByte[5] -= sbB;
        aDataSByte[6] = sbA;
        aDataSByte[6] *= sbB;
        aDataSByte[7] = sbA;
        aDataSByte[7] /= sbB;

        aDataSByte[8] = -sbA;
        // aDataSByte[9]  = Vector1<sbyte>::Exp(sbC);
        // aDataSByte[10] = Vector1<sbyte>::Ln(sbC);
        // aDataSByte[11] = Vector1<sbyte>::Sqrt(sbC);
        aDataSByte[12] = Vector1<sbyte>::Abs(sbA);
        aDataSByte[13] = Vector1<sbyte>::AbsDiff(sbA, sbB);

        /*aDataSByte[14] = sbC;
        aDataSByte[14].Exp();
        aDataSByte[15] = sbC;
        aDataSByte[15].Ln();
        aDataSByte[16] = sbC;
        aDataSByte[16].Sqrt();*/
        aDataSByte[17] = sbA;
        aDataSByte[17].Abs();
        aDataSByte[18] = sbA;
        aDataSByte[18].AbsDiff(sbB);

        aDataSByte[19] = Vector1<sbyte>::Min(sbA, sbB);
        aDataSByte[20] = Vector1<sbyte>::Max(sbA, sbB);

        aDataSByte[21] = sbA;
        aDataSByte[21].Min(sbB);
        aDataSByte[22] = sbA;
        aDataSByte[22].Max(sbB);

        aCompSbyte[0] = Vector1<sbyte>::CompareEQ(sbA, sbC);
        aCompSbyte[1] = Vector1<sbyte>::CompareLE(sbA, sbC);
        aCompSbyte[2] = Vector1<sbyte>::CompareLT(sbA, sbC);
        aCompSbyte[3] = Vector1<sbyte>::CompareGE(sbA, sbC);
        aCompSbyte[4] = Vector1<sbyte>::CompareGT(sbA, sbC);
        aCompSbyte[5] = Vector1<sbyte>::CompareNEQ(sbA, sbC);

        // ==
        sbA           = Vector1<sbyte>(120);
        sbB           = Vector1<sbyte>(120);
        aCompSbyte[6] = Vector1<byte>(byte(sbA == sbB));

        sbB           = Vector1<sbyte>(0);
        aCompSbyte[7] = Vector1<byte>(byte(sbA == sbB));

        sbB           = Vector1<sbyte>(120);
        aCompSbyte[8] = Vector1<byte>(byte(sbA == sbB));

        sbB           = Vector1<sbyte>(120);
        aCompSbyte[9] = Vector1<byte>(byte(sbA == sbB));

        sbB            = Vector1<sbyte>(10);
        aCompSbyte[10] = Vector1<byte>(byte(sbA == sbB));

        // <=
        sbA            = Vector1<sbyte>(120);
        sbB            = Vector1<sbyte>(120);
        aCompSbyte[11] = Vector1<byte>(byte(sbA <= sbB));

        sbB            = Vector1<sbyte>(110);
        aCompSbyte[12] = Vector1<byte>(byte(sbA <= sbB));

        sbB            = Vector1<sbyte>(121);
        aCompSbyte[13] = Vector1<byte>(byte(sbA <= sbB));

        sbB            = Vector1<sbyte>(120);
        aCompSbyte[14] = Vector1<byte>(byte(sbA <= sbB));

        sbB            = Vector1<sbyte>(120);
        aCompSbyte[15] = Vector1<byte>(byte(sbA <= sbB));

        // <
        sbA            = Vector1<sbyte>(120);
        sbB            = Vector1<sbyte>(120);
        aCompSbyte[16] = Vector1<byte>(byte(sbA < sbB));

        sbB            = Vector1<sbyte>(110);
        aCompSbyte[17] = Vector1<byte>(byte(sbA < sbB));

        sbB            = Vector1<sbyte>(121);
        aCompSbyte[18] = Vector1<byte>(byte(sbA < sbB));

        sbB            = Vector1<sbyte>(121);
        aCompSbyte[19] = Vector1<byte>(byte(sbA < sbB));

        sbB            = Vector1<sbyte>(121);
        aCompSbyte[20] = Vector1<byte>(byte(sbA < sbB));

        // >=
        sbA            = Vector1<sbyte>(120);
        sbB            = Vector1<sbyte>(120);
        aCompSbyte[21] = Vector1<byte>(byte(sbA >= sbB));

        sbB            = Vector1<sbyte>(110);
        aCompSbyte[22] = Vector1<byte>(byte(sbA >= sbB));

        sbB            = Vector1<sbyte>(110);
        aCompSbyte[23] = Vector1<byte>(byte(sbA >= sbB));

        sbB            = Vector1<sbyte>(110);
        aCompSbyte[24] = Vector1<byte>(byte(sbA >= sbB));

        sbB            = Vector1<sbyte>(127);
        aCompSbyte[25] = Vector1<byte>(byte(sbA >= sbB));

        // >
        sbA            = Vector1<sbyte>(120);
        sbB            = Vector1<sbyte>(120);
        aCompSbyte[26] = Vector1<byte>(byte(sbA > sbB));

        sbB            = Vector1<sbyte>(110);
        aCompSbyte[27] = Vector1<byte>(byte(sbA > sbB));

        sbB            = Vector1<sbyte>(121);
        aCompSbyte[28] = Vector1<byte>(byte(sbA > sbB));

        sbB            = Vector1<sbyte>(110);
        aCompSbyte[29] = Vector1<byte>(byte(sbA > sbB));

        sbB            = Vector1<sbyte>(127);
        aCompSbyte[30] = Vector1<byte>(byte(sbA > sbB));

        // !=
        sbA            = Vector1<sbyte>(120);
        sbB            = Vector1<sbyte>(120);
        aCompSbyte[31] = Vector1<byte>(byte(sbA != sbB));

        sbB            = Vector1<sbyte>(110);
        aCompSbyte[32] = Vector1<byte>(byte(sbA != sbB));

        sbB            = Vector1<sbyte>(120);
        aCompSbyte[33] = Vector1<byte>(byte(sbA != sbB));

        sbB            = Vector1<sbyte>(12);
        aCompSbyte[34] = Vector1<byte>(byte(sbA != sbB));

        sbB            = Vector1<sbyte>(120);
        aCompSbyte[35] = Vector1<byte>(byte(sbA != sbB));

        sbB            = Vector1<sbyte>(0);
        aCompSbyte[36] = Vector1<byte>(byte(sbA != sbB));
    }

    // byte
    if (blockIdx.x == 1)
    {
        Vector1<byte> bA(12);
        Vector1<byte> bB(120);
        Vector1<byte> bC(12);

        aDataByte[0] = bA + bB;
        aDataByte[1] = bA - bB;
        aDataByte[2] = bA * bB;
        aDataByte[3] = bA / bB;

        aDataByte[4] = bA;
        aDataByte[4] += bB;
        aDataByte[5] = bA;
        aDataByte[5] -= bB;
        aDataByte[6] = bA;
        aDataByte[6] *= bB;
        aDataByte[7] = bA;
        aDataByte[7] /= bB;

        // aDataByte[8] = -bA;
        // aDataByte[9]  = Vector1<byte>::Exp(bC);
        // aDataByte[10] = Vector1<byte>::Ln(bC);
        // aDataByte[11] = Vector1<byte>::Sqrt(bC);
        // aDataByte[12] = Vector1<byte>::Abs(bA);
        aDataByte[13] = Vector1<byte>::AbsDiff(bA, bB);

        /*aDataByte[14] = bC;
        aDataByte[14].Exp();
        aDataByte[15] = bC;
        aDataByte[15].Ln();
        aDataByte[16] = bC;
        aDataByte[16].Sqrt();
        aDataByte[17] = bA;
        aDataByte[17].Abs();*/
        aDataByte[18] = bA;
        aDataByte[18].AbsDiff(bB);

        aDataByte[19] = Vector1<byte>::Min(bA, bB);
        aDataByte[20] = Vector1<byte>::Max(bA, bB);

        aDataByte[21] = bA;
        aDataByte[21].Min(bB);
        aDataByte[22] = bA;
        aDataByte[22].Max(bB);

        aCompByte[0] = Vector1<byte>::CompareEQ(bA, bC);
        aCompByte[1] = Vector1<byte>::CompareLE(bA, bC);
        aCompByte[2] = Vector1<byte>::CompareLT(bA, bC);
        aCompByte[3] = Vector1<byte>::CompareGE(bA, bC);
        aCompByte[4] = Vector1<byte>::CompareGT(bA, bC);
        aCompByte[5] = Vector1<byte>::CompareNEQ(bA, bC);

        // ==
        bA           = Vector1<byte>(120);
        bB           = Vector1<byte>(120);
        aCompByte[6] = Vector1<byte>(byte(bA == bB));

        bB           = Vector1<byte>(0);
        aCompByte[7] = Vector1<byte>(byte(bA == bB));

        bB           = Vector1<byte>(120);
        aCompByte[8] = Vector1<byte>(byte(bA == bB));

        bB           = Vector1<byte>(120);
        aCompByte[9] = Vector1<byte>(byte(bA == bB));

        bB            = Vector1<byte>(110);
        aCompByte[10] = Vector1<byte>(byte(bA == bB));

        // <=
        bA            = Vector1<byte>(120);
        bB            = Vector1<byte>(120);
        aCompByte[11] = Vector1<byte>(byte(bA <= bB));

        bB            = Vector1<byte>(110);
        aCompByte[12] = Vector1<byte>(byte(bA <= bB));

        bB            = Vector1<byte>(121);
        aCompByte[13] = Vector1<byte>(byte(bA <= bB));

        bB            = Vector1<byte>(120);
        aCompByte[14] = Vector1<byte>(byte(bA <= bB));

        bB            = Vector1<byte>(120);
        aCompByte[15] = Vector1<byte>(byte(bA <= bB));

        // <
        bA            = Vector1<byte>(120);
        bB            = Vector1<byte>(120);
        aCompByte[16] = Vector1<byte>(byte(bA < bB));

        bB            = Vector1<byte>(110);
        aCompByte[17] = Vector1<byte>(byte(bA < bB));

        bB            = Vector1<byte>(121);
        aCompByte[18] = Vector1<byte>(byte(bA < bB));

        bB            = Vector1<byte>(121);
        aCompByte[19] = Vector1<byte>(byte(bA < bB));

        bB            = Vector1<byte>(121);
        aCompByte[20] = Vector1<byte>(byte(bA < bB));

        // >=
        bA            = Vector1<byte>(120);
        bB            = Vector1<byte>(120);
        aCompByte[21] = Vector1<byte>(byte(bA >= bB));

        bB            = Vector1<byte>(110);
        aCompByte[22] = Vector1<byte>(byte(bA >= bB));

        bB            = Vector1<byte>(110);
        aCompByte[23] = Vector1<byte>(byte(bA >= bB));

        bB            = Vector1<byte>(110);
        aCompByte[24] = Vector1<byte>(byte(bA >= bB));

        bB            = Vector1<byte>(127);
        aCompByte[25] = Vector1<byte>(byte(bA >= bB));

        // >
        bA            = Vector1<byte>(120);
        bB            = Vector1<byte>(120);
        aCompByte[26] = Vector1<byte>(byte(bA > bB));

        bB            = Vector1<byte>(110);
        aCompByte[27] = Vector1<byte>(byte(bA > bB));

        bB            = Vector1<byte>(121);
        aCompByte[28] = Vector1<byte>(byte(bA > bB));

        bB            = Vector1<byte>(121);
        aCompByte[29] = Vector1<byte>(byte(bA > bB));

        bB            = Vector1<byte>(110);
        aCompByte[30] = Vector1<byte>(byte(bA > bB));

        // !=
        bA            = Vector1<byte>(120);
        bB            = Vector1<byte>(120);
        aCompByte[31] = Vector1<byte>(byte(bA != bB));

        bB            = Vector1<byte>(110);
        aCompByte[32] = Vector1<byte>(byte(bA != bB));

        bB            = Vector1<byte>(120);
        aCompByte[33] = Vector1<byte>(byte(bA != bB));

        bB            = Vector1<byte>(120);
        aCompByte[34] = Vector1<byte>(byte(bA != bB));

        bB            = Vector1<byte>(120);
        aCompByte[35] = Vector1<byte>(byte(bA != bB));

        bB            = Vector1<byte>(0);
        aCompByte[36] = Vector1<byte>(byte(bA != bB));
    }

    // short
    if (blockIdx.x == 2)
    {
        Vector1<short> sA(12);
        Vector1<short> sB(120);
        Vector1<short> sC(12);

        aDataShort[0] = sA + sB;
        aDataShort[1] = sA - sB;
        aDataShort[2] = sA * sB;
        aDataShort[3] = sA / sB;

        aDataShort[4] = sA;
        aDataShort[4] += sB;
        aDataShort[5] = sA;
        aDataShort[5] -= sB;
        aDataShort[6] = sA;
        aDataShort[6] *= sB;
        aDataShort[7] = sA;
        aDataShort[7] /= sB;

        aDataShort[8] = -sA;
        // aDataShort[9]  = Vector1<short>::Exp(sC);
        // aDataShort[10] = Vector1<short>::Ln(sC);
        // aDataShort[11] = Vector1<short>::Sqrt(sC);
        aDataShort[12] = Vector1<short>::Abs(sA);
        aDataShort[13] = Vector1<short>::AbsDiff(sA, sB);

        /*aDataShort[14] = sC;
        aDataShort[14].Exp();
        aDataShort[15] = sC;
        aDataShort[15].Ln();
        aDataShort[16] = sC;
        aDataShort[16].Sqrt();*/
        aDataShort[17] = sA;
        aDataShort[17].Abs();
        aDataShort[18] = sA;
        aDataShort[18].AbsDiff(sB);

        aDataShort[19] = Vector1<short>::Min(sA, sB);
        aDataShort[20] = Vector1<short>::Max(sA, sB);

        aDataShort[21] = sA;
        aDataShort[21].Min(sB);
        aDataShort[22] = sA;
        aDataShort[22].Max(sB);

        aCompShort[0] = Vector1<short>::CompareEQ(sA, sC);
        aCompShort[1] = Vector1<short>::CompareLE(sA, sC);
        aCompShort[2] = Vector1<short>::CompareLT(sA, sC);
        aCompShort[3] = Vector1<short>::CompareGE(sA, sC);
        aCompShort[4] = Vector1<short>::CompareGT(sA, sC);
        aCompShort[5] = Vector1<short>::CompareNEQ(sA, sC);

        // ==
        sA            = Vector1<short>(120);
        sB            = Vector1<short>(120);
        aCompShort[6] = Vector1<byte>(byte(sA == sB));

        sB            = Vector1<short>(0);
        aCompShort[7] = Vector1<byte>(byte(sA == sB));

        sB            = Vector1<short>(120);
        aCompShort[8] = Vector1<byte>(byte(sA == sB));

        sB            = Vector1<short>(10);
        aCompShort[9] = Vector1<byte>(byte(sA == sB));

        sB             = Vector1<short>(120);
        aCompShort[10] = Vector1<byte>(byte(sA == sB));

        // <=
        sA             = Vector1<short>(120);
        sB             = Vector1<short>(120);
        aCompShort[11] = Vector1<byte>(byte(sA <= sB));

        sB             = Vector1<short>(110);
        aCompShort[12] = Vector1<byte>(byte(sA <= sB));

        sB             = Vector1<short>(121);
        aCompShort[13] = Vector1<byte>(byte(sA <= sB));

        sB             = Vector1<short>(120);
        aCompShort[14] = Vector1<byte>(byte(sA <= sB));

        sB             = Vector1<short>(120);
        aCompShort[15] = Vector1<byte>(byte(sA <= sB));

        // <
        sA             = Vector1<short>(120);
        sB             = Vector1<short>(120);
        aCompShort[16] = Vector1<byte>(byte(sA < sB));

        sB             = Vector1<short>(110);
        aCompShort[17] = Vector1<byte>(byte(sA < sB));

        sB             = Vector1<short>(121);
        aCompShort[18] = Vector1<byte>(byte(sA < sB));

        sB             = Vector1<short>(121);
        aCompShort[19] = Vector1<byte>(byte(sA < sB));

        sB             = Vector1<short>(121);
        aCompShort[20] = Vector1<byte>(byte(sA < sB));

        // >=
        sA             = Vector1<short>(120);
        sB             = Vector1<short>(120);
        aCompShort[21] = Vector1<byte>(byte(sA >= sB));

        sB             = Vector1<short>(110);
        aCompShort[22] = Vector1<byte>(byte(sA >= sB));

        sB             = Vector1<short>(110);
        aCompShort[23] = Vector1<byte>(byte(sA >= sB));

        sB             = Vector1<short>(110);
        aCompShort[24] = Vector1<byte>(byte(sA >= sB));

        sB             = Vector1<short>(127);
        aCompShort[25] = Vector1<byte>(byte(sA >= sB));

        // >
        sA             = Vector1<short>(120);
        sB             = Vector1<short>(120);
        aCompShort[26] = Vector1<byte>(byte(sA > sB));

        sB             = Vector1<short>(110);
        aCompShort[27] = Vector1<byte>(byte(sA > sB));

        sB             = Vector1<short>(121);
        aCompShort[28] = Vector1<byte>(byte(sA > sB));

        sB             = Vector1<short>(121);
        aCompShort[29] = Vector1<byte>(byte(sA > sB));

        sB             = Vector1<short>(119);
        aCompShort[30] = Vector1<byte>(byte(sA > sB));

        // !=
        sA             = Vector1<short>(120);
        sB             = Vector1<short>(120);
        aCompShort[31] = Vector1<byte>(byte(sA != sB));

        sB             = Vector1<short>(110);
        aCompShort[32] = Vector1<byte>(byte(sA != sB));

        sB             = Vector1<short>(120);
        aCompShort[33] = Vector1<byte>(byte(sA != sB));

        sB             = Vector1<short>(120);
        aCompShort[34] = Vector1<byte>(byte(sA != sB));

        sB             = Vector1<short>(120);
        aCompShort[35] = Vector1<byte>(byte(sA != sB));

        sB             = Vector1<short>(0);
        aCompShort[36] = Vector1<byte>(byte(sA != sB));
    }

    // ushort
    if (blockIdx.x == 3)
    {
        Vector1<ushort> usA(12);
        Vector1<ushort> usB(120);
        Vector1<ushort> usC(12);

        aDataUShort[0] = usA + usB;
        aDataUShort[1] = usA - usB;
        aDataUShort[2] = usA * usB;
        aDataUShort[3] = usA / usB;

        aDataUShort[4] = usA;
        aDataUShort[4] += usB;
        aDataUShort[5] = usA;
        aDataUShort[5] -= usB;
        aDataUShort[6] = usA;
        aDataUShort[6] *= usB;
        aDataUShort[7] = usA;
        aDataUShort[7] /= usB;

        // aDataUShort[8] = -usA;
        // aDataUShort[9]  = Vector1<ushort>::Exp(usC);
        // aDataUShort[10] = Vector1<ushort>::Ln(usC);
        // aDataUShort[11] = Vector1<ushort>::Sqrt(usC);
        // aDataUShort[12] = Vector1<ushort>::Abs(usA);
        aDataUShort[13] = Vector1<ushort>::AbsDiff(usA, usB);

        /*aDataUShort[14] = usC;
        aDataUShort[14].Exp();
        aDataUShort[15] = usC;
        aDataUShort[15].Ln();
        aDataUShort[16] = usC;
        aDataUShort[16].Sqrt();
        aDataUShort[17] = usA;
        aDataUShort[17].Abs();*/
        aDataUShort[18] = usA;
        aDataUShort[18].AbsDiff(usB);

        aDataUShort[19] = Vector1<ushort>::Min(usA, usB);
        aDataUShort[20] = Vector1<ushort>::Max(usA, usB);

        aDataUShort[21] = usA;
        aDataUShort[21].Min(usB);
        aDataUShort[22] = usA;
        aDataUShort[22].Max(usB);

        aCompUShort[0] = Vector1<ushort>::CompareEQ(usA, usC);
        aCompUShort[1] = Vector1<ushort>::CompareLE(usA, usC);
        aCompUShort[2] = Vector1<ushort>::CompareLT(usA, usC);
        aCompUShort[3] = Vector1<ushort>::CompareGE(usA, usC);
        aCompUShort[4] = Vector1<ushort>::CompareGT(usA, usC);
        aCompUShort[5] = Vector1<ushort>::CompareNEQ(usA, usC);

        // ==
        usA            = Vector1<ushort>(120);
        usB            = Vector1<ushort>(120);
        aCompUShort[6] = Vector1<byte>(byte(usA == usB));

        usB            = Vector1<ushort>(0);
        aCompUShort[7] = Vector1<byte>(byte(usA == usB));

        usB            = Vector1<ushort>(120);
        aCompUShort[8] = Vector1<byte>(byte(usA == usB));

        usB            = Vector1<ushort>(10);
        aCompUShort[9] = Vector1<byte>(byte(usA == usB));

        usB             = Vector1<ushort>(120);
        aCompUShort[10] = Vector1<byte>(byte(usA == usB));

        // <=
        usA             = Vector1<ushort>(120);
        usB             = Vector1<ushort>(120);
        aCompUShort[11] = Vector1<byte>(byte(usA <= usB));

        usB             = Vector1<ushort>(110);
        aCompUShort[12] = Vector1<byte>(byte(usA <= usB));

        usB             = Vector1<ushort>(121);
        aCompUShort[13] = Vector1<byte>(byte(usA <= usB));

        usB             = Vector1<ushort>(120);
        aCompUShort[14] = Vector1<byte>(byte(usA <= usB));

        usB             = Vector1<ushort>(120);
        aCompUShort[15] = Vector1<byte>(byte(usA <= usB));

        // <
        usA             = Vector1<ushort>(120);
        usB             = Vector1<ushort>(120);
        aCompUShort[16] = Vector1<byte>(byte(usA < usB));

        usB             = Vector1<ushort>(110);
        aCompUShort[17] = Vector1<byte>(byte(usA < usB));

        usB             = Vector1<ushort>(121);
        aCompUShort[18] = Vector1<byte>(byte(usA < usB));

        usB             = Vector1<ushort>(121);
        aCompUShort[19] = Vector1<byte>(byte(usA < usB));

        usB             = Vector1<ushort>(121);
        aCompUShort[20] = Vector1<byte>(byte(usA < usB));

        // >=
        usA             = Vector1<ushort>(120);
        usB             = Vector1<ushort>(120);
        aCompUShort[21] = Vector1<byte>(byte(usA >= usB));

        usB             = Vector1<ushort>(110);
        aCompUShort[22] = Vector1<byte>(byte(usA >= usB));

        usB             = Vector1<ushort>(110);
        aCompUShort[23] = Vector1<byte>(byte(usA >= usB));

        usB             = Vector1<ushort>(110);
        aCompUShort[24] = Vector1<byte>(byte(usA >= usB));

        usB             = Vector1<ushort>(121);
        aCompUShort[25] = Vector1<byte>(byte(usA >= usB));

        // >
        usA             = Vector1<ushort>(120);
        usB             = Vector1<ushort>(120);
        aCompUShort[26] = Vector1<byte>(byte(usA > usB));

        usB             = Vector1<ushort>(110);
        aCompUShort[27] = Vector1<byte>(byte(usA > usB));

        usB             = Vector1<ushort>(121);
        aCompUShort[28] = Vector1<byte>(byte(usA > usB));

        usB             = Vector1<ushort>(121);
        aCompUShort[29] = Vector1<byte>(byte(usA > usB));

        usB             = Vector1<ushort>(110);
        aCompUShort[30] = Vector1<byte>(byte(usA > usB));

        // !=
        usA             = Vector1<ushort>(120);
        usB             = Vector1<ushort>(120);
        aCompUShort[31] = Vector1<byte>(byte(usA != usB));

        usB             = Vector1<ushort>(110);
        aCompUShort[32] = Vector1<byte>(byte(usA != usB));

        usB             = Vector1<ushort>(120);
        aCompUShort[33] = Vector1<byte>(byte(usA != usB));

        usB             = Vector1<ushort>(10);
        aCompUShort[34] = Vector1<byte>(byte(usA != usB));

        usB             = Vector1<ushort>(120);
        aCompUShort[35] = Vector1<byte>(byte(usA != usB));

        usB             = Vector1<ushort>(0);
        aCompUShort[36] = Vector1<byte>(byte(usA != usB));
    }

    // BFloat
    if (blockIdx.x == 4)
    {
        Vector1<BFloat16> bfA(BFloat16(12.4f));
        Vector1<BFloat16> bfB(BFloat16(120.1f));
        Vector1<BFloat16> bfC(BFloat16(12.4f));

        aDataBFloat[0] = bfA + bfB;
        aDataBFloat[1] = bfA - bfB;
        aDataBFloat[2] = bfA * bfB;
        aDataBFloat[3] = bfA / bfB;

        aDataBFloat[4] = bfA;
        aDataBFloat[4] += bfB;
        aDataBFloat[5] = bfA;
        aDataBFloat[5] -= bfB;
        aDataBFloat[6] = bfA;
        aDataBFloat[6] *= bfB;
        aDataBFloat[7] = bfA;
        aDataBFloat[7] /= bfB;

        aDataBFloat[8]  = -bfA;
        aDataBFloat[9]  = Vector1<BFloat16>::Exp(bfC);
        aDataBFloat[10] = Vector1<BFloat16>::Ln(bfC);
        aDataBFloat[11] = Vector1<BFloat16>::Sqrt(bfC);
        aDataBFloat[12] = Vector1<BFloat16>::Abs(bfA);
        aDataBFloat[13] = Vector1<BFloat16>::AbsDiff(bfA, bfB);

        aDataBFloat[14] = bfC;
        aDataBFloat[14].Exp();
        aDataBFloat[15] = bfC;
        aDataBFloat[15].Ln();
        aDataBFloat[16] = bfC;
        aDataBFloat[16].Sqrt();
        aDataBFloat[17] = bfA;
        aDataBFloat[17].Abs();
        aDataBFloat[18] = bfA;
        aDataBFloat[18].AbsDiff(bfB);

        aDataBFloat[19] = Vector1<BFloat16>::Min(bfA, bfB);
        aDataBFloat[20] = Vector1<BFloat16>::Max(bfA, bfB);

        aDataBFloat[21] = bfA;
        aDataBFloat[21].Min(bfB);
        aDataBFloat[22] = bfA;
        aDataBFloat[22].Max(bfB);

        aDataBFloat[23] = bfA;
        aDataBFloat[23].Round();
        aDataBFloat[24] = bfA;
        aDataBFloat[24].Ceil();
        aDataBFloat[25] = bfA;
        aDataBFloat[25].Floor();
        aDataBFloat[26] = bfA;
        aDataBFloat[26].RoundNearest();
        aDataBFloat[27] = bfA;
        aDataBFloat[27].RoundZero();

        aDataBFloat[28] = Vector1<BFloat16>::Round(bfA);
        aDataBFloat[29] = Vector1<BFloat16>::Ceil(bfA);
        aDataBFloat[30] = Vector1<BFloat16>::Floor(bfA);
        aDataBFloat[31] = Vector1<BFloat16>::RoundNearest(bfA);
        aDataBFloat[32] = Vector1<BFloat16>::RoundZero(bfA);

        Vector1<float> bvec4(1.0f);
        aDataBFloat[33] = Vector1<BFloat16>(bvec4);
        Vector1<float> bvec42(aDataBFloat[33]);
        aDataBFloat[34] = Vector1<BFloat16>(bvec4 + bvec42);

        aCompBFloat[0] = Vector1<BFloat16>::CompareEQ(bfA, bfC);
        aCompBFloat[1] = Vector1<BFloat16>::CompareLE(bfA, bfC);
        aCompBFloat[2] = Vector1<BFloat16>::CompareLT(bfA, bfC);
        aCompBFloat[3] = Vector1<BFloat16>::CompareGE(bfA, bfC);
        aCompBFloat[4] = Vector1<BFloat16>::CompareGT(bfA, bfC);
        aCompBFloat[5] = Vector1<BFloat16>::CompareNEQ(bfA, bfC);

        // ==
        bfA            = Vector1<BFloat16>(BFloat16(120));
        bfB            = Vector1<BFloat16>(BFloat16(120));
        aCompBFloat[6] = Vector1<byte>(byte(bfA == bfB));

        bfB            = Vector1<BFloat16>(BFloat16(0));
        aCompBFloat[7] = Vector1<byte>(byte(bfA == bfB));

        bfB            = Vector1<BFloat16>(BFloat16(120));
        aCompBFloat[8] = Vector1<byte>(byte(bfA == bfB));

        bfB            = Vector1<BFloat16>(BFloat16(10));
        aCompBFloat[9] = Vector1<byte>(byte(bfA == bfB));

        bfB             = Vector1<BFloat16>(BFloat16(120));
        aCompBFloat[10] = Vector1<byte>(byte(bfA == bfB));

        // <=
        bfA             = Vector1<BFloat16>(BFloat16(120));
        bfB             = Vector1<BFloat16>(BFloat16(120));
        aCompBFloat[11] = Vector1<byte>(byte(bfA <= bfB));

        bfB             = Vector1<BFloat16>(BFloat16(110));
        aCompBFloat[12] = Vector1<byte>(byte(bfA <= bfB));

        bfB             = Vector1<BFloat16>(BFloat16(121));
        aCompBFloat[13] = Vector1<byte>(byte(bfA <= bfB));

        bfB             = Vector1<BFloat16>(BFloat16(120));
        aCompBFloat[14] = Vector1<byte>(byte(bfA <= bfB));

        bfB             = Vector1<BFloat16>(BFloat16(120));
        aCompBFloat[15] = Vector1<byte>(byte(bfA <= bfB));

        // <
        bfA             = Vector1<BFloat16>(BFloat16(120));
        bfB             = Vector1<BFloat16>(BFloat16(120));
        aCompBFloat[16] = Vector1<byte>(byte(bfA < bfB));

        bfB             = Vector1<BFloat16>(BFloat16(110));
        aCompBFloat[17] = Vector1<byte>(byte(bfA < bfB));

        bfB             = Vector1<BFloat16>(BFloat16(121));
        aCompBFloat[18] = Vector1<byte>(byte(bfA < bfB));

        bfB             = Vector1<BFloat16>(BFloat16(121));
        aCompBFloat[19] = Vector1<byte>(byte(bfA < bfB));

        bfB             = Vector1<BFloat16>(BFloat16(121));
        aCompBFloat[20] = Vector1<byte>(byte(bfA < bfB));

        // >=
        bfA             = Vector1<BFloat16>(BFloat16(120));
        bfB             = Vector1<BFloat16>(BFloat16(120));
        aCompBFloat[21] = Vector1<byte>(byte(bfA >= bfB));

        bfB             = Vector1<BFloat16>(BFloat16(110));
        aCompBFloat[22] = Vector1<byte>(byte(bfA >= bfB));

        bfB             = Vector1<BFloat16>(BFloat16(110));
        aCompBFloat[23] = Vector1<byte>(byte(bfA >= bfB));

        bfB             = Vector1<BFloat16>(BFloat16(110));
        aCompBFloat[24] = Vector1<byte>(byte(bfA >= bfB));

        bfB             = Vector1<BFloat16>(BFloat16(121));
        aCompBFloat[25] = Vector1<byte>(byte(bfA >= bfB));

        // >
        bfA             = Vector1<BFloat16>(BFloat16(120));
        bfB             = Vector1<BFloat16>(BFloat16(120));
        aCompBFloat[26] = Vector1<byte>(byte(bfA > bfB));

        bfB             = Vector1<BFloat16>(BFloat16(110));
        aCompBFloat[27] = Vector1<byte>(byte(bfA > bfB));

        bfB             = Vector1<BFloat16>(BFloat16(121));
        aCompBFloat[28] = Vector1<byte>(byte(bfA > bfB));

        bfB             = Vector1<BFloat16>(BFloat16(121));
        aCompBFloat[29] = Vector1<byte>(byte(bfA > bfB));

        bfB             = Vector1<BFloat16>(BFloat16(110));
        aCompBFloat[30] = Vector1<byte>(byte(bfA > bfB));

        // !=
        bfA             = Vector1<BFloat16>(BFloat16(120));
        bfB             = Vector1<BFloat16>(BFloat16(120));
        aCompBFloat[31] = Vector1<byte>(byte(bfA != bfB));

        bfB             = Vector1<BFloat16>(BFloat16(110));
        aCompBFloat[32] = Vector1<byte>(byte(bfA != bfB));

        bfB             = Vector1<BFloat16>(BFloat16(120));
        aCompBFloat[33] = Vector1<byte>(byte(bfA != bfB));

        bfB             = Vector1<BFloat16>(BFloat16(10));
        aCompBFloat[34] = Vector1<byte>(byte(bfA != bfB));

        bfB             = Vector1<BFloat16>(BFloat16(120));
        aCompBFloat[35] = Vector1<byte>(byte(bfA != bfB));

        bfB             = Vector1<BFloat16>(BFloat16(0));
        aCompBFloat[36] = Vector1<byte>(byte(bfA != bfB));
    }

    // Half
    if (blockIdx.x == 5)
    {
        Vector1<HalfFp16> hA(HalfFp16(12.4f));
        Vector1<HalfFp16> hB(HalfFp16(120.1f));
        Vector1<HalfFp16> hC(HalfFp16(12.4f));

        aDataHalf[0] = hA + hB;
        aDataHalf[1] = hA - hB;
        aDataHalf[2] = hA * hB;
        aDataHalf[3] = hA / hB;

        aDataHalf[4] = hA;
        aDataHalf[4] += hB;
        aDataHalf[5] = hA;
        aDataHalf[5] -= hB;
        aDataHalf[6] = hA;
        aDataHalf[6] *= hB;
        aDataHalf[7] = hA;
        aDataHalf[7] /= hB;

        aDataHalf[8]  = -hA;
        aDataHalf[9]  = Vector1<HalfFp16>::Exp(hC);
        aDataHalf[10] = Vector1<HalfFp16>::Ln(hC);
        aDataHalf[11] = Vector1<HalfFp16>::Sqrt(hC);
        aDataHalf[12] = Vector1<HalfFp16>::Abs(hA);
        aDataHalf[13] = Vector1<HalfFp16>::AbsDiff(hA, hB);

        aDataHalf[14] = hC;
        aDataHalf[14].Exp();
        aDataHalf[15] = hC;
        aDataHalf[15].Ln();
        aDataHalf[16] = hC;
        aDataHalf[16].Sqrt();
        aDataHalf[17] = hA;
        aDataHalf[17].Abs();
        aDataHalf[18] = hA;
        aDataHalf[18].AbsDiff(hB);

        aDataHalf[19] = Vector1<HalfFp16>::Min(hA, hB);
        aDataHalf[20] = Vector1<HalfFp16>::Max(hA, hB);

        aDataHalf[21] = hA;
        aDataHalf[21].Min(hB);
        aDataHalf[22] = hA;
        aDataHalf[22].Max(hB);

        aDataHalf[23] = hA;
        aDataHalf[23].Round();
        aDataHalf[24] = hA;
        aDataHalf[24].Ceil();
        aDataHalf[25] = hA;
        aDataHalf[25].Floor();
        aDataHalf[26] = hA;
        aDataHalf[26].RoundNearest();
        aDataHalf[27] = hA;
        aDataHalf[27].RoundZero();

        aDataHalf[28] = Vector1<HalfFp16>::Round(hA);
        aDataHalf[29] = Vector1<HalfFp16>::Ceil(hA);
        aDataHalf[30] = Vector1<HalfFp16>::Floor(hA);
        aDataHalf[31] = Vector1<HalfFp16>::RoundNearest(hA);
        aDataHalf[32] = Vector1<HalfFp16>::RoundZero(hA);

        Vector1<float> hvec4(1.0f);
        aDataHalf[33] = Vector1<HalfFp16>(hvec4);
        Vector1<float> hvec42(aDataHalf[33]);
        aDataHalf[34] = Vector1<HalfFp16>(hvec4 + hvec42);

        aCompHalf[0] = Vector1<HalfFp16>::CompareEQ(hA, hC);
        aCompHalf[1] = Vector1<HalfFp16>::CompareLE(hA, hC);
        aCompHalf[2] = Vector1<HalfFp16>::CompareLT(hA, hC);
        aCompHalf[3] = Vector1<HalfFp16>::CompareGE(hA, hC);
        aCompHalf[4] = Vector1<HalfFp16>::CompareGT(hA, hC);
        aCompHalf[5] = Vector1<HalfFp16>::CompareNEQ(hA, hC);

        // ==
        hA           = Vector1<HalfFp16>(HalfFp16(120));
        hB           = Vector1<HalfFp16>(HalfFp16(120));
        aCompHalf[6] = Vector1<byte>(byte(hA == hB));

        hB           = Vector1<HalfFp16>(HalfFp16(0));
        aCompHalf[7] = Vector1<byte>(byte(hA == hB));

        hB           = Vector1<HalfFp16>(HalfFp16(120));
        aCompHalf[8] = Vector1<byte>(byte(hA == hB));

        hB           = Vector1<HalfFp16>(HalfFp16(10));
        aCompHalf[9] = Vector1<byte>(byte(hA == hB));

        hB            = Vector1<HalfFp16>(HalfFp16(120));
        aCompHalf[10] = Vector1<byte>(byte(hA == hB));

        // <=
        hA            = Vector1<HalfFp16>(HalfFp16(120));
        hB            = Vector1<HalfFp16>(HalfFp16(120));
        aCompHalf[11] = Vector1<byte>(byte(hA <= hB));

        hB            = Vector1<HalfFp16>(HalfFp16(110));
        aCompHalf[12] = Vector1<byte>(byte(hA <= hB));

        hB            = Vector1<HalfFp16>(HalfFp16(121));
        aCompHalf[13] = Vector1<byte>(byte(hA <= hB));

        hB            = Vector1<HalfFp16>(HalfFp16(120));
        aCompHalf[14] = Vector1<byte>(byte(hA <= hB));

        hB            = Vector1<HalfFp16>(HalfFp16(120));
        aCompHalf[15] = Vector1<byte>(byte(hA <= hB));

        // <
        hA            = Vector1<HalfFp16>(HalfFp16(120));
        hB            = Vector1<HalfFp16>(HalfFp16(120));
        aCompHalf[16] = Vector1<byte>(byte(hA < hB));

        hB            = Vector1<HalfFp16>(HalfFp16(110));
        aCompHalf[17] = Vector1<byte>(byte(hA < hB));

        hB            = Vector1<HalfFp16>(HalfFp16(121));
        aCompHalf[18] = Vector1<byte>(byte(hA < hB));

        hB            = Vector1<HalfFp16>(HalfFp16(121));
        aCompHalf[19] = Vector1<byte>(byte(hA < hB));

        hB            = Vector1<HalfFp16>(HalfFp16(121));
        aCompHalf[20] = Vector1<byte>(byte(hA < hB));

        // >=
        hA            = Vector1<HalfFp16>(HalfFp16(120));
        hB            = Vector1<HalfFp16>(HalfFp16(120));
        aCompHalf[21] = Vector1<byte>(byte(hA >= hB));

        hB            = Vector1<HalfFp16>(HalfFp16(110));
        aCompHalf[22] = Vector1<byte>(byte(hA >= hB));

        hB            = Vector1<HalfFp16>(HalfFp16(110));
        aCompHalf[23] = Vector1<byte>(byte(hA >= hB));

        hB            = Vector1<HalfFp16>(HalfFp16(110));
        aCompHalf[24] = Vector1<byte>(byte(hA >= hB));

        hB            = Vector1<HalfFp16>(HalfFp16(127));
        aCompHalf[25] = Vector1<byte>(byte(hA >= hB));

        // >
        hA            = Vector1<HalfFp16>(HalfFp16(120));
        hB            = Vector1<HalfFp16>(HalfFp16(120));
        aCompHalf[26] = Vector1<byte>(byte(hA > hB));

        hB            = Vector1<HalfFp16>(HalfFp16(110));
        aCompHalf[27] = Vector1<byte>(byte(hA > hB));

        hB            = Vector1<HalfFp16>(HalfFp16(121));
        aCompHalf[28] = Vector1<byte>(byte(hA > hB));

        hB            = Vector1<HalfFp16>(HalfFp16(121));
        aCompHalf[29] = Vector1<byte>(byte(hA > hB));

        hB            = Vector1<HalfFp16>(HalfFp16(110));
        aCompHalf[30] = Vector1<byte>(byte(hA > hB));

        // !=
        hA            = Vector1<HalfFp16>(HalfFp16(120));
        hB            = Vector1<HalfFp16>(HalfFp16(120));
        aCompHalf[31] = Vector1<byte>(byte(hA != hB));

        hB            = Vector1<HalfFp16>(HalfFp16(110));
        aCompHalf[32] = Vector1<byte>(byte(hA != hB));

        hB            = Vector1<HalfFp16>(HalfFp16(120));
        aCompHalf[33] = Vector1<byte>(byte(hA != hB));

        hB            = Vector1<HalfFp16>(HalfFp16(10));
        aCompHalf[34] = Vector1<byte>(byte(hA != hB));

        hB            = Vector1<HalfFp16>(HalfFp16(120));
        aCompHalf[35] = Vector1<byte>(byte(hA != hB));

        hB            = Vector1<HalfFp16>(HalfFp16(0));
        aCompHalf[36] = Vector1<byte>(byte(hA != hB));
    }

    // Float
    if (blockIdx.x == 6)
    {
        Vector1<float> fA(12.4f);
        Vector1<float> fB(120.1f);
        Vector1<float> fC(12.4f);

        aDataFloat[0] = fA + fB;
        aDataFloat[1] = fA - fB;
        aDataFloat[2] = fA * fB;
        aDataFloat[3] = fA / fB;

        aDataFloat[4] = fA;
        aDataFloat[4] += fB;
        aDataFloat[5] = fA;
        aDataFloat[5] -= fB;
        aDataFloat[6] = fA;
        aDataFloat[6] *= fB;
        aDataFloat[7] = fA;
        aDataFloat[7] /= fB;

        aDataFloat[8]  = -fA;
        aDataFloat[9]  = Vector1<float>::Exp(fC);
        aDataFloat[10] = Vector1<float>::Ln(fC);
        aDataFloat[11] = Vector1<float>::Sqrt(fC);
        aDataFloat[12] = Vector1<float>::Abs(fA);
        aDataFloat[13] = Vector1<float>::AbsDiff(fA, fB);

        aDataFloat[14] = fC;
        aDataFloat[14].Exp();
        aDataFloat[15] = fC;
        aDataFloat[15].Ln();
        aDataFloat[16] = fC;
        aDataFloat[16].Sqrt();
        aDataFloat[17] = fA;
        aDataFloat[17].Abs();
        aDataFloat[18] = fA;
        aDataFloat[18].AbsDiff(fB);

        aDataFloat[19] = Vector1<float>::Min(fA, fB);
        aDataFloat[20] = Vector1<float>::Max(fA, fB);

        aDataFloat[21] = fA;
        aDataFloat[21].Min(fB);
        aDataFloat[22] = fA;
        aDataFloat[22].Max(fB);

        aDataFloat[23] = fA;
        aDataFloat[23].Round();
        aDataFloat[24] = fA;
        aDataFloat[24].Ceil();
        aDataFloat[25] = fA;
        aDataFloat[25].Floor();
        aDataFloat[26] = fA;
        aDataFloat[26].RoundNearest();
        aDataFloat[27] = fA;
        aDataFloat[27].RoundZero();

        aDataFloat[28] = Vector1<float>::Round(fA);
        aDataFloat[29] = Vector1<float>::Ceil(fA);
        aDataFloat[30] = Vector1<float>::Floor(fA);
        aDataFloat[31] = Vector1<float>::RoundNearest(fA);
        aDataFloat[32] = Vector1<float>::RoundZero(fA);

        Vector1<float> bvec4(1.0f);
        aDataFloat[33] = Vector1<HalfFp16>(bvec4);
        Vector1<float> bvec42(aDataFloat[33]);
        aDataFloat[34] = Vector1<HalfFp16>(bvec4 + bvec42);

        aCompFloat[0] = Vector1<float>::CompareEQ(fA, fC);
        aCompFloat[1] = Vector1<float>::CompareLE(fA, fC);
        aCompFloat[2] = Vector1<float>::CompareLT(fA, fC);
        aCompFloat[3] = Vector1<float>::CompareGE(fA, fC);
        aCompFloat[4] = Vector1<float>::CompareGT(fA, fC);
        aCompFloat[5] = Vector1<float>::CompareNEQ(fA, fC);

        // ==
        fA            = Vector1<float>(120);
        fB            = Vector1<float>(120);
        aCompFloat[6] = Vector1<byte>(byte(fA == fB));

        fB            = Vector1<float>(0.0f);
        aCompFloat[7] = Vector1<byte>(byte(fA == fB));

        fB            = Vector1<float>(120);
        aCompFloat[8] = Vector1<byte>(byte(fA == fB));

        fB            = Vector1<float>(10);
        aCompFloat[9] = Vector1<byte>(byte(fA == fB));

        fB             = Vector1<float>(120);
        aCompFloat[10] = Vector1<byte>(byte(fA == fB));

        // <=
        fA             = Vector1<float>(120);
        fB             = Vector1<float>(120);
        aCompFloat[11] = Vector1<byte>(byte(fA <= fB));

        fB             = Vector1<float>(110);
        aCompFloat[12] = Vector1<byte>(byte(fA <= fB));

        fB             = Vector1<float>(121);
        aCompFloat[13] = Vector1<byte>(byte(fA <= fB));

        fB             = Vector1<float>(120);
        aCompFloat[14] = Vector1<byte>(byte(fA <= fB));

        fB             = Vector1<float>(120);
        aCompFloat[15] = Vector1<byte>(byte(fA <= fB));

        // <
        fA             = Vector1<float>(120);
        fB             = Vector1<float>(120);
        aCompFloat[16] = Vector1<byte>(byte(fA < fB));

        fB             = Vector1<float>(110);
        aCompFloat[17] = Vector1<byte>(byte(fA < fB));

        fB             = Vector1<float>(121);
        aCompFloat[18] = Vector1<byte>(byte(fA < fB));

        fB             = Vector1<float>(121);
        aCompFloat[19] = Vector1<byte>(byte(fA < fB));

        fB             = Vector1<float>(121);
        aCompFloat[20] = Vector1<byte>(byte(fA < fB));

        // >=
        fA             = Vector1<float>(120);
        fB             = Vector1<float>(120);
        aCompFloat[21] = Vector1<byte>(byte(fA >= fB));

        fB             = Vector1<float>(110);
        aCompFloat[22] = Vector1<byte>(byte(fA >= fB));

        fB             = Vector1<float>(110);
        aCompFloat[23] = Vector1<byte>(byte(fA >= fB));

        fB             = Vector1<float>(110);
        aCompFloat[24] = Vector1<byte>(byte(fA >= fB));

        fB             = Vector1<float>(120);
        aCompFloat[25] = Vector1<byte>(byte(fA >= fB));

        // >
        fA             = Vector1<float>(120);
        fB             = Vector1<float>(120);
        aCompFloat[26] = Vector1<byte>(byte(fA > fB));

        fB             = Vector1<float>(110);
        aCompFloat[27] = Vector1<byte>(byte(fA > fB));

        fB             = Vector1<float>(121);
        aCompFloat[28] = Vector1<byte>(byte(fA > fB));

        fB             = Vector1<float>(121);
        aCompFloat[29] = Vector1<byte>(byte(fA > fB));

        fB             = Vector1<float>(110);
        aCompFloat[30] = Vector1<byte>(byte(fA > fB));

        // !=
        fA             = Vector1<float>(120);
        fB             = Vector1<float>(120);
        aCompFloat[31] = Vector1<byte>(byte(fA != fB));

        fB             = Vector1<float>(110);
        aCompFloat[32] = Vector1<byte>(byte(fA != fB));

        fB             = Vector1<float>(120);
        aCompFloat[33] = Vector1<byte>(byte(fA != fB));

        fB             = Vector1<float>(10);
        aCompFloat[34] = Vector1<byte>(byte(fA != fB));

        fB             = Vector1<float>(120);
        aCompFloat[35] = Vector1<byte>(byte(fA != fB));

        fB             = Vector1<float>(0.0f);
        aCompFloat[36] = Vector1<byte>(byte(fA != fB));
    }
}

void runtest_vector1_kernel(Vector1<byte> *aDataByte, Vector1<byte> *aCompByte,         //
                            Vector1<sbyte> *aDataSByte, Vector1<byte> *aCompSbyte,      //
                            Vector1<short> *aDataShort, Vector1<byte> *aCompShort,      //
                            Vector1<ushort> *aDataUShort, Vector1<byte> *aCompUShort,   //
                            Vector1<BFloat16> *aDataBFloat, Vector1<byte> *aCompBFloat, //
                            Vector1<HalfFp16> *aDataHalf, Vector1<byte> *aCompHalf,     //
                            Vector1<float> *aDataFloat, Vector1<byte> *aCompFloat)
{
    test_vector1_kernel<<<7, 1>>>(aDataByte, aCompByte,     //
                                  aDataSByte, aCompSbyte,   //
                                  aDataShort, aCompShort,   //
                                  aDataUShort, aCompUShort, //
                                  aDataBFloat, aCompBFloat, //
                                  aDataHalf, aCompHalf,     //
                                  aDataFloat, aCompFloat);
}
} // namespace cuda
} // namespace opp