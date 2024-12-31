#include <cfloat>
#include <cmath>
#include <common/bfloat16.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/vector3.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace opp
{
namespace cuda
{

__global__ void test_vector3_kernel(Vector3<byte> *aDataByte, Vector3<byte> *aCompByte,         //
                                    Vector3<sbyte> *aDataSByte, Vector3<byte> *aCompSbyte,      //
                                    Vector3<short> *aDataShort, Vector3<byte> *aCompShort,      //
                                    Vector3<ushort> *aDataUShort, Vector3<byte> *aCompUShort,   //
                                    Vector3<BFloat16> *aDataBFloat, Vector3<byte> *aCompBFloat, //
                                    Vector3<HalfFp16> *aDataHalf, Vector3<byte> *aCompHalf,     //
                                    Vector3<float> *aDataFloat, Vector3<byte> *aCompFloat)
{
    if (threadIdx.x > 0 || blockIdx.x > 6)
    {
        return;
    }

    // sbyte
    if (blockIdx.x == 0)
    {
        Vector3<sbyte> sbA(12, -120, -10);
        Vector3<sbyte> sbB(120, -80, 30);
        Vector3<sbyte> sbC(12, 20, 0);

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
        // aDataSByte[9]  = Vector3<sbyte>::Exp(sbC);
        // aDataSByte[10] = Vector3<sbyte>::Ln(sbC);
        // aDataSByte[11] = Vector3<sbyte>::Sqrt(sbC);
        aDataSByte[12] = Vector3<sbyte>::Abs(sbA);
        // aDataSByte[13] = Vector3<sbyte>::AbsDiff(sbA, sbB);

        /*aDataSByte[14] = sbC;
        aDataSByte[14].Exp();
        aDataSByte[15] = sbC;
        aDataSByte[15].Ln();
        aDataSByte[16] = sbC;
        aDataSByte[16].Sqrt();*/
        aDataSByte[17] = sbA;
        aDataSByte[17].Abs();
        // aDataSByte[18] = sbA;
        // aDataSByte[18].AbsDiff(sbB);

        aDataSByte[19] = Vector3<sbyte>::Min(sbA, sbB);
        aDataSByte[20] = Vector3<sbyte>::Max(sbA, sbB);

        aDataSByte[21] = sbA;
        aDataSByte[21].Min(sbB);
        aDataSByte[22] = sbA;
        aDataSByte[22].Max(sbB);

        aCompSbyte[0] = Vector3<sbyte>::CompareEQ(sbA, sbC);
        aCompSbyte[1] = Vector3<sbyte>::CompareLE(sbA, sbC);
        aCompSbyte[2] = Vector3<sbyte>::CompareLT(sbA, sbC);
        aCompSbyte[3] = Vector3<sbyte>::CompareGE(sbA, sbC);
        aCompSbyte[4] = Vector3<sbyte>::CompareGT(sbA, sbC);
        aCompSbyte[5] = Vector3<sbyte>::CompareNEQ(sbA, sbC);

        // ==
        sbA           = Vector3<sbyte>(120, -120, -10);
        sbB           = Vector3<sbyte>(120, -120, -10);
        aCompSbyte[6] = Vector3<byte>(byte(sbA == sbB));

        sbB           = Vector3<sbyte>(0, -120, -10);
        aCompSbyte[7] = Vector3<byte>(byte(sbA == sbB));

        sbB           = Vector3<sbyte>(120, 0, -10);
        aCompSbyte[8] = Vector3<byte>(byte(sbA == sbB));

        sbB           = Vector3<sbyte>(120, -120, 0);
        aCompSbyte[9] = Vector3<byte>(byte(sbA == sbB));

        sbB            = Vector3<sbyte>(120, -120, -100);
        aCompSbyte[10] = Vector3<byte>(byte(sbA == sbB));

        // <=
        sbA            = Vector3<sbyte>(120, -120, -10);
        sbB            = Vector3<sbyte>(120, -120, -10);
        aCompSbyte[11] = Vector3<byte>(byte(sbA <= sbB));

        sbB            = Vector3<sbyte>(110, -120, -10);
        aCompSbyte[12] = Vector3<byte>(byte(sbA <= sbB));

        sbB            = Vector3<sbyte>(121, 0, -10);
        aCompSbyte[13] = Vector3<byte>(byte(sbA <= sbB));

        sbB            = Vector3<sbyte>(120, -120, 0);
        aCompSbyte[14] = Vector3<byte>(byte(sbA <= sbB));

        sbB            = Vector3<sbyte>(120, -120, 100);
        aCompSbyte[15] = Vector3<byte>(byte(sbA <= sbB));

        // <
        sbA            = Vector3<sbyte>(120, -120, -10);
        sbB            = Vector3<sbyte>(120, -120, -10);
        aCompSbyte[16] = Vector3<byte>(byte(sbA < sbB));

        sbB            = Vector3<sbyte>(110, -110, 0);
        aCompSbyte[17] = Vector3<byte>(byte(sbA < sbB));

        sbB            = Vector3<sbyte>(121, 0, 0);
        aCompSbyte[18] = Vector3<byte>(byte(sbA < sbB));

        sbB            = Vector3<sbyte>(121, -121, 0);
        aCompSbyte[19] = Vector3<byte>(byte(sbA < sbB));

        sbB            = Vector3<sbyte>(121, 0, 0);
        aCompSbyte[20] = Vector3<byte>(byte(sbA < sbB));

        // >=
        sbA            = Vector3<sbyte>(120, -120, -10);
        sbB            = Vector3<sbyte>(120, -120, -10);
        aCompSbyte[21] = Vector3<byte>(byte(sbA >= sbB));

        sbB            = Vector3<sbyte>(110, -121, -10);
        aCompSbyte[22] = Vector3<byte>(byte(sbA >= sbB));

        sbB            = Vector3<sbyte>(110, 0, -20);
        aCompSbyte[23] = Vector3<byte>(byte(sbA >= sbB));

        sbB            = Vector3<sbyte>(110, -120, -20);
        aCompSbyte[24] = Vector3<byte>(byte(sbA >= sbB));

        sbB            = Vector3<sbyte>(120, -120, 100);
        aCompSbyte[25] = Vector3<byte>(byte(sbA >= sbB));

        // >
        sbA            = Vector3<sbyte>(120, -120, -10);
        sbB            = Vector3<sbyte>(120, -120, -10);
        aCompSbyte[26] = Vector3<byte>(byte(sbA > sbB));

        sbB            = Vector3<sbyte>(110, -121, -20);
        aCompSbyte[27] = Vector3<byte>(byte(sbA > sbB));

        sbB            = Vector3<sbyte>(121, 0, -10);
        aCompSbyte[28] = Vector3<byte>(byte(sbA > sbB));

        sbB            = Vector3<sbyte>(110, -121, 0);
        aCompSbyte[29] = Vector3<byte>(byte(sbA > sbB));

        sbB            = Vector3<sbyte>(110, -121, 0);
        aCompSbyte[30] = Vector3<byte>(byte(sbA > sbB));

        // !=
        sbA            = Vector3<sbyte>(120, -120, -10);
        sbB            = Vector3<sbyte>(120, -120, -10);
        aCompSbyte[31] = Vector3<byte>(byte(sbA != sbB));

        sbB            = Vector3<sbyte>(110, -120, -10);
        aCompSbyte[32] = Vector3<byte>(byte(sbA != sbB));

        sbB            = Vector3<sbyte>(120, 0, -10);
        aCompSbyte[33] = Vector3<byte>(byte(sbA != sbB));

        sbB            = Vector3<sbyte>(120, -120, 0);
        aCompSbyte[34] = Vector3<byte>(byte(sbA != sbB));

        sbB            = Vector3<sbyte>(120, 0, 0);
        aCompSbyte[35] = Vector3<byte>(byte(sbA != sbB));

        sbB            = Vector3<sbyte>(0, 0, 0);
        aCompSbyte[36] = Vector3<byte>(byte(sbA != sbB));
    }

    // byte
    if (blockIdx.x == 1)
    {
        Vector3<byte> bA(12, 120, 100);
        Vector3<byte> bB(120, 180, 30);
        Vector3<byte> bC(12, 20, 100);

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
        // aDataByte[9]  = Vector3<byte>::Exp(bC);
        // aDataByte[10] = Vector3<byte>::Ln(bC);
        // aDataByte[11] = Vector3<byte>::Sqrt(bC);
        // aDataByte[12] = Vector3<byte>::Abs(bA);
        // aDataByte[13] = Vector3<byte>::AbsDiff(bA, bB);

        /*aDataByte[14] = bC;
        aDataByte[14].Exp();
        aDataByte[15] = bC;
        aDataByte[15].Ln();
        aDataByte[16] = bC;
        aDataByte[16].Sqrt();
        aDataByte[17] = bA;
        aDataByte[17].Abs();*/
        // aDataByte[18] = bA;
        // aDataByte[18].AbsDiff(bB);

        aDataByte[19] = Vector3<byte>::Min(bA, bB);
        aDataByte[20] = Vector3<byte>::Max(bA, bB);

        aDataByte[21] = bA;
        aDataByte[21].Min(bB);
        aDataByte[22] = bA;
        aDataByte[22].Max(bB);

        aCompByte[0] = Vector3<byte>::CompareEQ(bA, bC);
        aCompByte[1] = Vector3<byte>::CompareLE(bA, bC);
        aCompByte[2] = Vector3<byte>::CompareLT(bA, bC);
        aCompByte[3] = Vector3<byte>::CompareGE(bA, bC);
        aCompByte[4] = Vector3<byte>::CompareGT(bA, bC);
        aCompByte[5] = Vector3<byte>::CompareNEQ(bA, bC);

        // ==
        bA           = Vector3<byte>(120, 200, 10);
        bB           = Vector3<byte>(120, 200, 10);
        aCompByte[6] = Vector3<byte>(byte(bA == bB));

        bB           = Vector3<byte>(0, 200, 10);
        aCompByte[7] = Vector3<byte>(byte(bA == bB));

        bB           = Vector3<byte>(120, 0, 10);
        aCompByte[8] = Vector3<byte>(byte(bA == bB));

        bB           = Vector3<byte>(120, 200, 20);
        aCompByte[9] = Vector3<byte>(byte(bA == bB));

        bB            = Vector3<byte>(120, 200, 100);
        aCompByte[10] = Vector3<byte>(byte(bA == bB));

        // <=
        bA            = Vector3<byte>(120, 200, 10);
        bB            = Vector3<byte>(120, 200, 10);
        aCompByte[11] = Vector3<byte>(byte(bA <= bB));

        bB            = Vector3<byte>(110, 200, 10);
        aCompByte[12] = Vector3<byte>(byte(bA <= bB));

        bB            = Vector3<byte>(121, 220, 10);
        aCompByte[13] = Vector3<byte>(byte(bA <= bB));

        bB            = Vector3<byte>(120, 200, 20);
        aCompByte[14] = Vector3<byte>(byte(bA <= bB));

        bB            = Vector3<byte>(120, 200, 100);
        aCompByte[15] = Vector3<byte>(byte(bA <= bB));

        // <
        bA            = Vector3<byte>(120, 200, 10);
        bB            = Vector3<byte>(120, 200, 10);
        aCompByte[16] = Vector3<byte>(byte(bA < bB));

        bB            = Vector3<byte>(110, 220, 20);
        aCompByte[17] = Vector3<byte>(byte(bA < bB));

        bB            = Vector3<byte>(121, 0, 20);
        aCompByte[18] = Vector3<byte>(byte(bA < bB));

        bB            = Vector3<byte>(121, 220, 20);
        aCompByte[19] = Vector3<byte>(byte(bA < bB));

        bB            = Vector3<byte>(121, 220, 20);
        aCompByte[20] = Vector3<byte>(byte(bA < bB));

        // >=
        bA            = Vector3<byte>(120, 200, 10);
        bB            = Vector3<byte>(120, 200, 10);
        aCompByte[21] = Vector3<byte>(byte(bA >= bB));

        bB            = Vector3<byte>(110, 0, 10);
        aCompByte[22] = Vector3<byte>(byte(bA >= bB));

        bB            = Vector3<byte>(110, 220, 0);
        aCompByte[23] = Vector3<byte>(byte(bA >= bB));

        bB            = Vector3<byte>(110, 200, 0);
        aCompByte[24] = Vector3<byte>(byte(bA >= bB));

        bB            = Vector3<byte>(120, 200, 100);
        aCompByte[25] = Vector3<byte>(byte(bA >= bB));

        // >
        bA            = Vector3<byte>(120, 200, 10);
        bB            = Vector3<byte>(120, 200, 10);
        aCompByte[26] = Vector3<byte>(byte(bA > bB));

        bB            = Vector3<byte>(110, 0, 0);
        aCompByte[27] = Vector3<byte>(byte(bA > bB));

        bB            = Vector3<byte>(121, 220, 10);
        aCompByte[28] = Vector3<byte>(byte(bA > bB));

        bB            = Vector3<byte>(110, 0, 20);
        aCompByte[29] = Vector3<byte>(byte(bA > bB));

        bB            = Vector3<byte>(110, 220, 20);
        aCompByte[30] = Vector3<byte>(byte(bA > bB));

        // !=
        bA            = Vector3<byte>(120, 200, 10);
        bB            = Vector3<byte>(120, 200, 10);
        aCompByte[31] = Vector3<byte>(byte(bA != bB));

        bB            = Vector3<byte>(110, 200, 10);
        aCompByte[32] = Vector3<byte>(byte(bA != bB));

        bB            = Vector3<byte>(120, 220, 10);
        aCompByte[33] = Vector3<byte>(byte(bA != bB));

        bB            = Vector3<byte>(120, 200, 0);
        aCompByte[34] = Vector3<byte>(byte(bA != bB));

        bB            = Vector3<byte>(120, 0, 0);
        aCompByte[35] = Vector3<byte>(byte(bA != bB));

        bB            = Vector3<byte>(0, 0, 0);
        aCompByte[36] = Vector3<byte>(byte(bA != bB));
    }

    // short
    if (blockIdx.x == 2)
    {
        Vector3<short> sA(12, -30000, -10);
        Vector3<short> sB(120, -3000, 30);
        Vector3<short> sC(12, -30000, 0);

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
        // aDataShort[9]  = Vector3<short>::Exp(sC);
        // aDataShort[10] = Vector3<short>::Ln(sC);
        // aDataShort[11] = Vector3<short>::Sqrt(sC);
        aDataShort[12] = Vector3<short>::Abs(sA);
        // aDataShort[13] = Vector3<short>::AbsDiff(sA, sB);

        /*aDataShort[14] = sC;
        aDataShort[14].Exp();
        aDataShort[15] = sC;
        aDataShort[15].Ln();
        aDataShort[16] = sC;
        aDataShort[16].Sqrt();*/
        aDataShort[17] = sA;
        aDataShort[17].Abs();
        // aDataShort[18] = sA;
        // aDataShort[18].AbsDiff(sB);

        aDataShort[19] = Vector3<short>::Min(sA, sB);
        aDataShort[20] = Vector3<short>::Max(sA, sB);

        aDataShort[21] = sA;
        aDataShort[21].Min(sB);
        aDataShort[22] = sA;
        aDataShort[22].Max(sB);

        aCompShort[0] = Vector3<short>::CompareEQ(sA, sC);
        aCompShort[1] = Vector3<short>::CompareLE(sA, sC);
        aCompShort[2] = Vector3<short>::CompareLT(sA, sC);
        aCompShort[3] = Vector3<short>::CompareGE(sA, sC);
        aCompShort[4] = Vector3<short>::CompareGT(sA, sC);
        aCompShort[5] = Vector3<short>::CompareNEQ(sA, sC);

        // ==
        sA            = Vector3<short>(120, -120, -10);
        sB            = Vector3<short>(120, -120, -10);
        aCompShort[6] = Vector3<byte>(byte(sA == sB));

        sB            = Vector3<short>(0, -120, -10);
        aCompShort[7] = Vector3<byte>(byte(sA == sB));

        sB            = Vector3<short>(120, 0, -10);
        aCompShort[8] = Vector3<byte>(byte(sA == sB));

        sB            = Vector3<short>(120, -120, 0);
        aCompShort[9] = Vector3<byte>(byte(sA == sB));

        sB             = Vector3<short>(120, -120, -100);
        aCompShort[10] = Vector3<byte>(byte(sA == sB));

        // <=
        sA             = Vector3<short>(120, -120, -10);
        sB             = Vector3<short>(120, -120, -10);
        aCompShort[11] = Vector3<byte>(byte(sA <= sB));

        sB             = Vector3<short>(110, -120, -10);
        aCompShort[12] = Vector3<byte>(byte(sA <= sB));

        sB             = Vector3<short>(121, 0, -10);
        aCompShort[13] = Vector3<byte>(byte(sA <= sB));

        sB             = Vector3<short>(120, -120, 0);
        aCompShort[14] = Vector3<byte>(byte(sA <= sB));

        sB             = Vector3<short>(120, -120, 100);
        aCompShort[15] = Vector3<byte>(byte(sA <= sB));

        // <
        sA             = Vector3<short>(120, -120, -10);
        sB             = Vector3<short>(120, -120, -10);
        aCompShort[16] = Vector3<byte>(byte(sA < sB));

        sB             = Vector3<short>(110, -110, 0);
        aCompShort[17] = Vector3<byte>(byte(sA < sB));

        sB             = Vector3<short>(121, 0, 0);
        aCompShort[18] = Vector3<byte>(byte(sA < sB));

        sB             = Vector3<short>(121, -121, 0);
        aCompShort[19] = Vector3<byte>(byte(sA < sB));

        sB             = Vector3<short>(121, 0, 0);
        aCompShort[20] = Vector3<byte>(byte(sA < sB));

        // >=
        sA             = Vector3<short>(120, -120, -10);
        sB             = Vector3<short>(120, -120, -10);
        aCompShort[21] = Vector3<byte>(byte(sA >= sB));

        sB             = Vector3<short>(110, -121, -10);
        aCompShort[22] = Vector3<byte>(byte(sA >= sB));

        sB             = Vector3<short>(110, 0, -20);
        aCompShort[23] = Vector3<byte>(byte(sA >= sB));

        sB             = Vector3<short>(110, -120, -20);
        aCompShort[24] = Vector3<byte>(byte(sA >= sB));

        sB             = Vector3<short>(120, -120, 100);
        aCompShort[25] = Vector3<byte>(byte(sA >= sB));

        // >
        sA             = Vector3<short>(120, -120, -10);
        sB             = Vector3<short>(120, -120, -10);
        aCompShort[26] = Vector3<byte>(byte(sA > sB));

        sB             = Vector3<short>(110, -121, -20);
        aCompShort[27] = Vector3<byte>(byte(sA > sB));

        sB             = Vector3<short>(121, 0, -10);
        aCompShort[28] = Vector3<byte>(byte(sA > sB));

        sB             = Vector3<short>(110, -121, 0);
        aCompShort[29] = Vector3<byte>(byte(sA > sB));

        sB             = Vector3<short>(110, -121, 0);
        aCompShort[30] = Vector3<byte>(byte(sA > sB));

        // !=
        sA             = Vector3<short>(120, -120, -10);
        sB             = Vector3<short>(120, -120, -10);
        aCompShort[31] = Vector3<byte>(byte(sA != sB));

        sB             = Vector3<short>(110, -120, -10);
        aCompShort[32] = Vector3<byte>(byte(sA != sB));

        sB             = Vector3<short>(120, 0, -10);
        aCompShort[33] = Vector3<byte>(byte(sA != sB));

        sB             = Vector3<short>(120, -120, 0);
        aCompShort[34] = Vector3<byte>(byte(sA != sB));

        sB             = Vector3<short>(120, 0, 0);
        aCompShort[35] = Vector3<byte>(byte(sA != sB));

        sB             = Vector3<short>(0, 0, 0);
        aCompShort[36] = Vector3<byte>(byte(sA != sB));
    }

    // ushort
    if (blockIdx.x == 3)
    {
        Vector3<ushort> usA(12, 60000, 100);
        Vector3<ushort> usB(120, 7000, 30);
        Vector3<ushort> usC(12, 20, 100);

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
        // aDataUShort[9]  = Vector3<ushort>::Exp(usC);
        // aDataUShort[10] = Vector3<ushort>::Ln(usC);
        // aDataUShort[11] = Vector3<ushort>::Sqrt(usC);
        // aDataUShort[12] = Vector3<ushort>::Abs(usA);
        // aDataUShort[13] = Vector3<ushort>::AbsDiff(usA, usB);

        /*aDataUShort[14] = usC;
        aDataUShort[14].Exp();
        aDataUShort[15] = usC;
        aDataUShort[15].Ln();
        aDataUShort[16] = usC;
        aDataUShort[16].Sqrt();
        aDataUShort[17] = usA;
        aDataUShort[17].Abs();*/
        // aDataUShort[18] = usA;
        // aDataUShort[18].AbsDiff(usB);

        aDataUShort[19] = Vector3<ushort>::Min(usA, usB);
        aDataUShort[20] = Vector3<ushort>::Max(usA, usB);

        aDataUShort[21] = usA;
        aDataUShort[21].Min(usB);
        aDataUShort[22] = usA;
        aDataUShort[22].Max(usB);

        aCompUShort[0] = Vector3<ushort>::CompareEQ(usA, usC);
        aCompUShort[1] = Vector3<ushort>::CompareLE(usA, usC);
        aCompUShort[2] = Vector3<ushort>::CompareLT(usA, usC);
        aCompUShort[3] = Vector3<ushort>::CompareGE(usA, usC);
        aCompUShort[4] = Vector3<ushort>::CompareGT(usA, usC);
        aCompUShort[5] = Vector3<ushort>::CompareNEQ(usA, usC);

        // ==
        usA            = Vector3<ushort>(120, 200, 10);
        usB            = Vector3<ushort>(120, 200, 10);
        aCompUShort[6] = Vector3<byte>(byte(usA == usB));

        usB            = Vector3<ushort>(0, 200, 10);
        aCompUShort[7] = Vector3<byte>(byte(usA == usB));

        usB            = Vector3<ushort>(120, 0, 10);
        aCompUShort[8] = Vector3<byte>(byte(usA == usB));

        usB            = Vector3<ushort>(120, 200, 20);
        aCompUShort[9] = Vector3<byte>(byte(usA == usB));

        usB             = Vector3<ushort>(120, 200, 100);
        aCompUShort[10] = Vector3<byte>(byte(usA == usB));

        // <=
        usA             = Vector3<ushort>(120, 200, 10);
        usB             = Vector3<ushort>(120, 200, 10);
        aCompUShort[11] = Vector3<byte>(byte(usA <= usB));

        usB             = Vector3<ushort>(110, 200, 10);
        aCompUShort[12] = Vector3<byte>(byte(usA <= usB));

        usB             = Vector3<ushort>(121, 220, 10);
        aCompUShort[13] = Vector3<byte>(byte(usA <= usB));

        usB             = Vector3<ushort>(120, 200, 20);
        aCompUShort[14] = Vector3<byte>(byte(usA <= usB));

        usB             = Vector3<ushort>(120, 200, 100);
        aCompUShort[15] = Vector3<byte>(byte(usA <= usB));

        // <
        usA             = Vector3<ushort>(120, 200, 10);
        usB             = Vector3<ushort>(120, 200, 10);
        aCompUShort[16] = Vector3<byte>(byte(usA < usB));

        usB             = Vector3<ushort>(110, 220, 20);
        aCompUShort[17] = Vector3<byte>(byte(usA < usB));

        usB             = Vector3<ushort>(121, 0, 20);
        aCompUShort[18] = Vector3<byte>(byte(usA < usB));

        usB             = Vector3<ushort>(121, 220, 20);
        aCompUShort[19] = Vector3<byte>(byte(usA < usB));

        usB             = Vector3<ushort>(121, 220, 20);
        aCompUShort[20] = Vector3<byte>(byte(usA < usB));

        // >=
        usA             = Vector3<ushort>(120, 200, 10);
        usB             = Vector3<ushort>(120, 200, 10);
        aCompUShort[21] = Vector3<byte>(byte(usA >= usB));

        usB             = Vector3<ushort>(110, 0, 10);
        aCompUShort[22] = Vector3<byte>(byte(usA >= usB));

        usB             = Vector3<ushort>(110, 220, 0);
        aCompUShort[23] = Vector3<byte>(byte(usA >= usB));

        usB             = Vector3<ushort>(110, 200, 0);
        aCompUShort[24] = Vector3<byte>(byte(usA >= usB));

        usB             = Vector3<ushort>(120, 200, 100);
        aCompUShort[25] = Vector3<byte>(byte(usA >= usB));

        // >
        usA             = Vector3<ushort>(120, 200, 10);
        usB             = Vector3<ushort>(120, 200, 10);
        aCompUShort[26] = Vector3<byte>(byte(usA > usB));

        usB             = Vector3<ushort>(110, 0, 0);
        aCompUShort[27] = Vector3<byte>(byte(usA > usB));

        usB             = Vector3<ushort>(121, 220, 10);
        aCompUShort[28] = Vector3<byte>(byte(usA > usB));

        usB             = Vector3<ushort>(110, 0, 20);
        aCompUShort[29] = Vector3<byte>(byte(usA > usB));

        usB             = Vector3<ushort>(110, 220, 20);
        aCompUShort[30] = Vector3<byte>(byte(usA > usB));

        // !=
        usA             = Vector3<ushort>(120, 200, 10);
        usB             = Vector3<ushort>(120, 200, 10);
        aCompUShort[31] = Vector3<byte>(byte(usA != usB));

        usB             = Vector3<ushort>(110, 200, 10);
        aCompUShort[32] = Vector3<byte>(byte(usA != usB));

        usB             = Vector3<ushort>(120, 220, 10);
        aCompUShort[33] = Vector3<byte>(byte(usA != usB));

        usB             = Vector3<ushort>(120, 200, 0);
        aCompUShort[34] = Vector3<byte>(byte(usA != usB));

        usB             = Vector3<ushort>(120, 0, 0);
        aCompUShort[35] = Vector3<byte>(byte(usA != usB));

        usB             = Vector3<ushort>(0, 0, 0);
        aCompUShort[36] = Vector3<byte>(byte(usA != usB));
    }

    // BFloat
    if (blockIdx.x == 4)
    {
        Vector3<BFloat16> bfA(BFloat16(12.4f), BFloat16(-30000.2f), BFloat16(-10.5f));
        Vector3<BFloat16> bfB(BFloat16(120.1f), BFloat16(-3000.1f), BFloat16(30.2f));
        Vector3<BFloat16> bfC(BFloat16(12.4f), BFloat16(-30000.2f), BFloat16(0.0f));

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
        aDataBFloat[9]  = Vector3<BFloat16>::Exp(bfC);
        aDataBFloat[10] = Vector3<BFloat16>::Ln(bfC);
        aDataBFloat[11] = Vector3<BFloat16>::Sqrt(bfC);
        aDataBFloat[12] = Vector3<BFloat16>::Abs(bfA);
        aDataBFloat[13] = Vector3<BFloat16>::AbsDiff(bfA, bfB);

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

        aDataBFloat[19] = Vector3<BFloat16>::Min(bfA, bfB);
        aDataBFloat[20] = Vector3<BFloat16>::Max(bfA, bfB);

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

        aDataBFloat[28] = Vector3<BFloat16>::Round(bfA);
        aDataBFloat[29] = Vector3<BFloat16>::Ceil(bfA);
        aDataBFloat[30] = Vector3<BFloat16>::Floor(bfA);
        aDataBFloat[31] = Vector3<BFloat16>::RoundNearest(bfA);
        aDataBFloat[32] = Vector3<BFloat16>::RoundZero(bfA);

        Vector3<float> bvec4(1.0f, 2.0f, 3.0f);
        aDataBFloat[33] = Vector3<BFloat16>(bvec4);
        Vector3<float> bvec42(aDataBFloat[33]);
        aDataBFloat[34] = Vector3<BFloat16>(bvec4 + bvec42);

        aCompBFloat[0] = Vector3<BFloat16>::CompareEQ(bfA, bfC);
        aCompBFloat[1] = Vector3<BFloat16>::CompareLE(bfA, bfC);
        aCompBFloat[2] = Vector3<BFloat16>::CompareLT(bfA, bfC);
        aCompBFloat[3] = Vector3<BFloat16>::CompareGE(bfA, bfC);
        aCompBFloat[4] = Vector3<BFloat16>::CompareGT(bfA, bfC);
        aCompBFloat[5] = Vector3<BFloat16>::CompareNEQ(bfA, bfC);

        // ==
        bfA            = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(-10));
        bfB            = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(-10));
        aCompBFloat[6] = Vector3<byte>(byte(bfA == bfB));

        bfB            = Vector3<BFloat16>(BFloat16(0), BFloat16(-120), BFloat16(-10));
        aCompBFloat[7] = Vector3<byte>(byte(bfA == bfB));

        bfB            = Vector3<BFloat16>(BFloat16(120), BFloat16(0), BFloat16(-10));
        aCompBFloat[8] = Vector3<byte>(byte(bfA == bfB));

        bfB            = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(0));
        aCompBFloat[9] = Vector3<byte>(byte(bfA == bfB));

        bfB             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(-100));
        aCompBFloat[10] = Vector3<byte>(byte(bfA == bfB));

        // <=
        bfA             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(-10));
        bfB             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(-10));
        aCompBFloat[11] = Vector3<byte>(byte(bfA <= bfB));

        bfB             = Vector3<BFloat16>(BFloat16(110), BFloat16(-120), BFloat16(-10));
        aCompBFloat[12] = Vector3<byte>(byte(bfA <= bfB));

        bfB             = Vector3<BFloat16>(BFloat16(121), BFloat16(0), BFloat16(-10));
        aCompBFloat[13] = Vector3<byte>(byte(bfA <= bfB));

        bfB             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(0));
        aCompBFloat[14] = Vector3<byte>(byte(bfA <= bfB));

        bfB             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(100));
        aCompBFloat[15] = Vector3<byte>(byte(bfA <= bfB));

        // <
        bfA             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(-10));
        bfB             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(-10));
        aCompBFloat[16] = Vector3<byte>(byte(bfA < bfB));

        bfB             = Vector3<BFloat16>(BFloat16(110), BFloat16(-110), BFloat16(0));
        aCompBFloat[17] = Vector3<byte>(byte(bfA < bfB));

        bfB             = Vector3<BFloat16>(BFloat16(121), BFloat16(0), BFloat16(0));
        aCompBFloat[18] = Vector3<byte>(byte(bfA < bfB));

        bfB             = Vector3<BFloat16>(BFloat16(121), BFloat16(-121), BFloat16(0));
        aCompBFloat[19] = Vector3<byte>(byte(bfA < bfB));

        bfB             = Vector3<BFloat16>(BFloat16(121), BFloat16(0), BFloat16(0));
        aCompBFloat[20] = Vector3<byte>(byte(bfA < bfB));

        // >=
        bfA             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(-10));
        bfB             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(-10));
        aCompBFloat[21] = Vector3<byte>(byte(bfA >= bfB));

        bfB             = Vector3<BFloat16>(BFloat16(110), BFloat16(-121), BFloat16(-10));
        aCompBFloat[22] = Vector3<byte>(byte(bfA >= bfB));

        bfB             = Vector3<BFloat16>(BFloat16(110), BFloat16(0), BFloat16(-20));
        aCompBFloat[23] = Vector3<byte>(byte(bfA >= bfB));

        bfB             = Vector3<BFloat16>(BFloat16(110), BFloat16(-120), BFloat16(-20));
        aCompBFloat[24] = Vector3<byte>(byte(bfA >= bfB));

        bfB             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(100));
        aCompBFloat[25] = Vector3<byte>(byte(bfA >= bfB));

        // >
        bfA             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(-10));
        bfB             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(-10));
        aCompBFloat[26] = Vector3<byte>(byte(bfA > bfB));

        bfB             = Vector3<BFloat16>(BFloat16(110), BFloat16(-121), BFloat16(-20));
        aCompBFloat[27] = Vector3<byte>(byte(bfA > bfB));

        bfB             = Vector3<BFloat16>(BFloat16(121), BFloat16(0), BFloat16(-10));
        aCompBFloat[28] = Vector3<byte>(byte(bfA > bfB));

        bfB             = Vector3<BFloat16>(BFloat16(110), BFloat16(-121), BFloat16(0));
        aCompBFloat[29] = Vector3<byte>(byte(bfA > bfB));

        bfB             = Vector3<BFloat16>(BFloat16(110), BFloat16(-121), BFloat16(0));
        aCompBFloat[30] = Vector3<byte>(byte(bfA > bfB));

        // !=
        bfA             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(-10));
        bfB             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(-10));
        aCompBFloat[31] = Vector3<byte>(byte(bfA != bfB));

        bfB             = Vector3<BFloat16>(BFloat16(110), BFloat16(-120), BFloat16(-10));
        aCompBFloat[32] = Vector3<byte>(byte(bfA != bfB));

        bfB             = Vector3<BFloat16>(BFloat16(120), BFloat16(0), BFloat16(-10));
        aCompBFloat[33] = Vector3<byte>(byte(bfA != bfB));

        bfB             = Vector3<BFloat16>(BFloat16(120), BFloat16(-120), BFloat16(0));
        aCompBFloat[34] = Vector3<byte>(byte(bfA != bfB));

        bfB             = Vector3<BFloat16>(BFloat16(120), BFloat16(0), BFloat16(0));
        aCompBFloat[35] = Vector3<byte>(byte(bfA != bfB));

        bfB             = Vector3<BFloat16>(BFloat16(0), BFloat16(0), BFloat16(0));
        aCompBFloat[36] = Vector3<byte>(byte(bfA != bfB));
    }

    // Half
    if (blockIdx.x == 5)
    {
        Vector3<HalfFp16> hA(HalfFp16(12.4f), HalfFp16(-30000.2f), HalfFp16(-10.5f));
        Vector3<HalfFp16> hB(HalfFp16(120.1f), HalfFp16(-3000.1f), HalfFp16(30.2f));
        Vector3<HalfFp16> hC(HalfFp16(12.4f), HalfFp16(-30000.2f), HalfFp16(0.0f));

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
        aDataHalf[9]  = Vector3<HalfFp16>::Exp(hC);
        aDataHalf[10] = Vector3<HalfFp16>::Ln(hC);
        aDataHalf[11] = Vector3<HalfFp16>::Sqrt(hC);
        aDataHalf[12] = Vector3<HalfFp16>::Abs(hA);
        aDataHalf[13] = Vector3<HalfFp16>::AbsDiff(hA, hB);

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

        aDataHalf[19] = Vector3<HalfFp16>::Min(hA, hB);
        aDataHalf[20] = Vector3<HalfFp16>::Max(hA, hB);

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

        aDataHalf[28] = Vector3<HalfFp16>::Round(hA);
        aDataHalf[29] = Vector3<HalfFp16>::Ceil(hA);
        aDataHalf[30] = Vector3<HalfFp16>::Floor(hA);
        aDataHalf[31] = Vector3<HalfFp16>::RoundNearest(hA);
        aDataHalf[32] = Vector3<HalfFp16>::RoundZero(hA);

        Vector3<float> hvec4(1.0f, 2.0f, 3.0f);
        aDataHalf[33] = Vector3<HalfFp16>(hvec4);
        Vector3<float> hvec42(aDataHalf[33]);
        aDataHalf[34] = Vector3<HalfFp16>(hvec4 + hvec42);

        aCompHalf[0] = Vector3<HalfFp16>::CompareEQ(hA, hC);
        aCompHalf[1] = Vector3<HalfFp16>::CompareLE(hA, hC);
        aCompHalf[2] = Vector3<HalfFp16>::CompareLT(hA, hC);
        aCompHalf[3] = Vector3<HalfFp16>::CompareGE(hA, hC);
        aCompHalf[4] = Vector3<HalfFp16>::CompareGT(hA, hC);
        aCompHalf[5] = Vector3<HalfFp16>::CompareNEQ(hA, hC);

        // ==
        hA           = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(-10));
        hB           = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(-10));
        aCompHalf[6] = Vector3<byte>(byte(hA == hB));

        hB           = Vector3<HalfFp16>(HalfFp16(0), HalfFp16(-120), HalfFp16(-10));
        aCompHalf[7] = Vector3<byte>(byte(hA == hB));

        hB           = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(0), HalfFp16(-10));
        aCompHalf[8] = Vector3<byte>(byte(hA == hB));

        hB           = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(0));
        aCompHalf[9] = Vector3<byte>(byte(hA == hB));

        hB            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(-100));
        aCompHalf[10] = Vector3<byte>(byte(hA == hB));

        // <=
        hA            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(-10));
        hB            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(-10));
        aCompHalf[11] = Vector3<byte>(byte(hA <= hB));

        hB            = Vector3<HalfFp16>(HalfFp16(110), HalfFp16(-120), HalfFp16(-10));
        aCompHalf[12] = Vector3<byte>(byte(hA <= hB));

        hB            = Vector3<HalfFp16>(HalfFp16(121), HalfFp16(0), HalfFp16(-10));
        aCompHalf[13] = Vector3<byte>(byte(hA <= hB));

        hB            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(0));
        aCompHalf[14] = Vector3<byte>(byte(hA <= hB));

        hB            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(100));
        aCompHalf[15] = Vector3<byte>(byte(hA <= hB));

        // <
        hA            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(-10));
        hB            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(-10));
        aCompHalf[16] = Vector3<byte>(byte(hA < hB));

        hB            = Vector3<HalfFp16>(HalfFp16(110), HalfFp16(-110), HalfFp16(0));
        aCompHalf[17] = Vector3<byte>(byte(hA < hB));

        hB            = Vector3<HalfFp16>(HalfFp16(121), HalfFp16(0), HalfFp16(0));
        aCompHalf[18] = Vector3<byte>(byte(hA < hB));

        hB            = Vector3<HalfFp16>(HalfFp16(121), HalfFp16(-121), HalfFp16(0));
        aCompHalf[19] = Vector3<byte>(byte(hA < hB));

        hB            = Vector3<HalfFp16>(HalfFp16(121), HalfFp16(0), HalfFp16(0));
        aCompHalf[20] = Vector3<byte>(byte(hA < hB));

        // >=
        hA            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(-10));
        hB            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(-10));
        aCompHalf[21] = Vector3<byte>(byte(hA >= hB));

        hB            = Vector3<HalfFp16>(HalfFp16(110), HalfFp16(-121), HalfFp16(-10));
        aCompHalf[22] = Vector3<byte>(byte(hA >= hB));

        hB            = Vector3<HalfFp16>(HalfFp16(110), HalfFp16(0), HalfFp16(-20));
        aCompHalf[23] = Vector3<byte>(byte(hA >= hB));

        hB            = Vector3<HalfFp16>(HalfFp16(110), HalfFp16(-120), HalfFp16(-20));
        aCompHalf[24] = Vector3<byte>(byte(hA >= hB));

        hB            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(100));
        aCompHalf[25] = Vector3<byte>(byte(hA >= hB));

        // >
        hA            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(-10));
        hB            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(-10));
        aCompHalf[26] = Vector3<byte>(byte(hA > hB));

        hB            = Vector3<HalfFp16>(HalfFp16(110), HalfFp16(-121), HalfFp16(-20));
        aCompHalf[27] = Vector3<byte>(byte(hA > hB));

        hB            = Vector3<HalfFp16>(HalfFp16(121), HalfFp16(0), HalfFp16(-10));
        aCompHalf[28] = Vector3<byte>(byte(hA > hB));

        hB            = Vector3<HalfFp16>(HalfFp16(110), HalfFp16(-121), HalfFp16(0));
        aCompHalf[29] = Vector3<byte>(byte(hA > hB));

        hB            = Vector3<HalfFp16>(HalfFp16(110), HalfFp16(-121), HalfFp16(0));
        aCompHalf[30] = Vector3<byte>(byte(hA > hB));

        // !=
        hA            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(-10));
        hB            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(-10));
        aCompHalf[31] = Vector3<byte>(byte(hA != hB));

        hB            = Vector3<HalfFp16>(HalfFp16(110), HalfFp16(-120), HalfFp16(-10));
        aCompHalf[32] = Vector3<byte>(byte(hA != hB));

        hB            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(0), HalfFp16(-10));
        aCompHalf[33] = Vector3<byte>(byte(hA != hB));

        hB            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(-120), HalfFp16(0));
        aCompHalf[34] = Vector3<byte>(byte(hA != hB));

        hB            = Vector3<HalfFp16>(HalfFp16(120), HalfFp16(0), HalfFp16(0));
        aCompHalf[35] = Vector3<byte>(byte(hA != hB));

        hB            = Vector3<HalfFp16>(HalfFp16(0), HalfFp16(0), HalfFp16(0));
        aCompHalf[36] = Vector3<byte>(byte(hA != hB));
    }

    // Float
    if (blockIdx.x == 6)
    {
        Vector3<float> fA(12.4f, -30000.2f, -10.5f);
        Vector3<float> fB(120.1f, -3000.1f, 30.2f);
        Vector3<float> fC(12.4f, -30000.2f, 0.0f);

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
        aDataFloat[9]  = Vector3<float>::Exp(fC);
        aDataFloat[10] = Vector3<float>::Ln(fC);
        aDataFloat[11] = Vector3<float>::Sqrt(fC);
        aDataFloat[12] = Vector3<float>::Abs(fA);
        aDataFloat[13] = Vector3<float>::AbsDiff(fA, fB);

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

        aDataFloat[19] = Vector3<float>::Min(fA, fB);
        aDataFloat[20] = Vector3<float>::Max(fA, fB);

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

        aDataFloat[28] = Vector3<float>::Round(fA);
        aDataFloat[29] = Vector3<float>::Ceil(fA);
        aDataFloat[30] = Vector3<float>::Floor(fA);
        aDataFloat[31] = Vector3<float>::RoundNearest(fA);
        aDataFloat[32] = Vector3<float>::RoundZero(fA);

        Vector3<float> bvec4(1.0f, 2.0f, 3.0f);
        aDataFloat[33] = Vector3<HalfFp16>(bvec4);
        Vector3<float> bvec42(aDataFloat[33]);
        aDataFloat[34] = Vector3<HalfFp16>(bvec4 + bvec42);

        aCompFloat[0] = Vector3<float>::CompareEQ(fA, fC);
        aCompFloat[1] = Vector3<float>::CompareLE(fA, fC);
        aCompFloat[2] = Vector3<float>::CompareLT(fA, fC);
        aCompFloat[3] = Vector3<float>::CompareGE(fA, fC);
        aCompFloat[4] = Vector3<float>::CompareGT(fA, fC);
        aCompFloat[5] = Vector3<float>::CompareNEQ(fA, fC);

        // ==
        fA            = Vector3<float>(120, -120, -10);
        fB            = Vector3<float>(120, -120, -10);
        aCompFloat[6] = Vector3<byte>(byte(fA == fB));

        fB            = Vector3<float>(0, -120, -10);
        aCompFloat[7] = Vector3<byte>(byte(fA == fB));

        fB            = Vector3<float>(120, 0, -10);
        aCompFloat[8] = Vector3<byte>(byte(fA == fB));

        fB            = Vector3<float>(120, -120, 0);
        aCompFloat[9] = Vector3<byte>(byte(fA == fB));

        fB             = Vector3<float>(120, -120, -100);
        aCompFloat[10] = Vector3<byte>(byte(fA == fB));

        // <=
        fA             = Vector3<float>(120, -120, -10);
        fB             = Vector3<float>(120, -120, -10);
        aCompFloat[11] = Vector3<byte>(byte(fA <= fB));

        fB             = Vector3<float>(110, -120, -10);
        aCompFloat[12] = Vector3<byte>(byte(fA <= fB));

        fB             = Vector3<float>(121, 0, -10);
        aCompFloat[13] = Vector3<byte>(byte(fA <= fB));

        fB             = Vector3<float>(120, -120, 0);
        aCompFloat[14] = Vector3<byte>(byte(fA <= fB));

        fB             = Vector3<float>(120, -120, 100);
        aCompFloat[15] = Vector3<byte>(byte(fA <= fB));

        // <
        fA             = Vector3<float>(120, -120, -10);
        fB             = Vector3<float>(120, -120, -10);
        aCompFloat[16] = Vector3<byte>(byte(fA < fB));

        fB             = Vector3<float>(110, -110, 0);
        aCompFloat[17] = Vector3<byte>(byte(fA < fB));

        fB             = Vector3<float>(121, 0, 0);
        aCompFloat[18] = Vector3<byte>(byte(fA < fB));

        fB             = Vector3<float>(121, -121, 0);
        aCompFloat[19] = Vector3<byte>(byte(fA < fB));

        fB             = Vector3<float>(121, 0, 0);
        aCompFloat[20] = Vector3<byte>(byte(fA < fB));

        // >=
        fA             = Vector3<float>(120, -120, -10);
        fB             = Vector3<float>(120, -120, -10);
        aCompFloat[21] = Vector3<byte>(byte(fA >= fB));

        fB             = Vector3<float>(110, -121, -10);
        aCompFloat[22] = Vector3<byte>(byte(fA >= fB));

        fB             = Vector3<float>(110, 0, -20);
        aCompFloat[23] = Vector3<byte>(byte(fA >= fB));

        fB             = Vector3<float>(110, -120, -20);
        aCompFloat[24] = Vector3<byte>(byte(fA >= fB));

        fB             = Vector3<float>(120, -120, 100);
        aCompFloat[25] = Vector3<byte>(byte(fA >= fB));

        // >
        fA             = Vector3<float>(120, -120, -10);
        fB             = Vector3<float>(120, -120, -10);
        aCompFloat[26] = Vector3<byte>(byte(fA > fB));

        fB             = Vector3<float>(110, -121, -20);
        aCompFloat[27] = Vector3<byte>(byte(fA > fB));

        fB             = Vector3<float>(121, 0, -10);
        aCompFloat[28] = Vector3<byte>(byte(fA > fB));

        fB             = Vector3<float>(110, -121, 0);
        aCompFloat[29] = Vector3<byte>(byte(fA > fB));

        fB             = Vector3<float>(110, -121, 0);
        aCompFloat[30] = Vector3<byte>(byte(fA > fB));

        // !=
        fA             = Vector3<float>(120, -120, -10);
        fB             = Vector3<float>(120, -120, -10);
        aCompFloat[31] = Vector3<byte>(byte(fA != fB));

        fB             = Vector3<float>(110, -120, -10);
        aCompFloat[32] = Vector3<byte>(byte(fA != fB));

        fB             = Vector3<float>(120, 0, -10);
        aCompFloat[33] = Vector3<byte>(byte(fA != fB));

        fB             = Vector3<float>(120, -120, 0);
        aCompFloat[34] = Vector3<byte>(byte(fA != fB));

        fB             = Vector3<float>(120, 0, 0);
        aCompFloat[35] = Vector3<byte>(byte(fA != fB));

        fB             = Vector3<float>(0, 0, 0);
        aCompFloat[36] = Vector3<byte>(byte(fA != fB));
    }
}

void runtest_vector3_kernel(Vector3<byte> *aDataByte, Vector3<byte> *aCompByte,         //
                            Vector3<sbyte> *aDataSByte, Vector3<byte> *aCompSbyte,      //
                            Vector3<short> *aDataShort, Vector3<byte> *aCompShort,      //
                            Vector3<ushort> *aDataUShort, Vector3<byte> *aCompUShort,   //
                            Vector3<BFloat16> *aDataBFloat, Vector3<byte> *aCompBFloat, //
                            Vector3<HalfFp16> *aDataHalf, Vector3<byte> *aCompHalf,     //
                            Vector3<float> *aDataFloat, Vector3<byte> *aCompFloat)
{
    test_vector3_kernel<<<7, 1>>>(aDataByte, aCompByte,     //
                                  aDataSByte, aCompSbyte,   //
                                  aDataShort, aCompShort,   //
                                  aDataUShort, aCompUShort, //
                                  aDataBFloat, aCompBFloat, //
                                  aDataHalf, aCompHalf,     //
                                  aDataFloat, aCompFloat);
}
} // namespace cuda
} // namespace opp