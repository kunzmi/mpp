#include <cfloat>
#include <cmath>
#include <common/bfloat16.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/vector2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace opp
{
namespace cuda
{

__global__ void test_vector2_kernel(Vector2<byte> *aDataByte, Vector2<byte> *aCompByte,         //
                                    Vector2<sbyte> *aDataSByte, Vector2<byte> *aCompSbyte,      //
                                    Vector2<short> *aDataShort, Vector2<byte> *aCompShort,      //
                                    Vector2<ushort> *aDataUShort, Vector2<byte> *aCompUShort,   //
                                    Vector2<BFloat16> *aDataBFloat, Vector2<byte> *aCompBFloat, //
                                    Vector2<HalfFp16> *aDataHalf, Vector2<byte> *aCompHalf,     //
                                    Vector2<float> *aDataFloat, Vector2<byte> *aCompFloat)
{
    if (threadIdx.x > 0 || blockIdx.x > 6)
    {
        return;
    }

    // sbyte
    if (blockIdx.x == 0)
    {
        Vector2<sbyte> sbA(12, -120);
        Vector2<sbyte> sbB(120, -80);
        Vector2<sbyte> sbC(12, 20);

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
        // aDataSByte[9]  = Vector2<sbyte>::Exp(sbC);
        // aDataSByte[10] = Vector2<sbyte>::Ln(sbC);
        // aDataSByte[11] = Vector2<sbyte>::Sqrt(sbC);
        aDataSByte[12] = Vector2<sbyte>::Abs(sbA);
        aDataSByte[13] = Vector2<sbyte>::AbsDiff(sbA, sbB);

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

        aDataSByte[19] = Vector2<sbyte>::Min(sbA, sbB);
        aDataSByte[20] = Vector2<sbyte>::Max(sbA, sbB);

        aDataSByte[21] = sbA;
        aDataSByte[21].Min(sbB);
        aDataSByte[22] = sbA;
        aDataSByte[22].Max(sbB);

        aCompSbyte[0] = Vector2<sbyte>::CompareEQ(sbA, sbC);
        aCompSbyte[1] = Vector2<sbyte>::CompareLE(sbA, sbC);
        aCompSbyte[2] = Vector2<sbyte>::CompareLT(sbA, sbC);
        aCompSbyte[3] = Vector2<sbyte>::CompareGE(sbA, sbC);
        aCompSbyte[4] = Vector2<sbyte>::CompareGT(sbA, sbC);
        aCompSbyte[5] = Vector2<sbyte>::CompareNEQ(sbA, sbC);

        // ==
        sbA           = Vector2<sbyte>(120, -120);
        sbB           = Vector2<sbyte>(120, -120);
        aCompSbyte[6] = Vector2<byte>(byte(sbA == sbB));

        sbB           = Vector2<sbyte>(0, -120);
        aCompSbyte[7] = Vector2<byte>(byte(sbA == sbB));

        sbB           = Vector2<sbyte>(120, 0);
        aCompSbyte[8] = Vector2<byte>(byte(sbA == sbB));

        sbB           = Vector2<sbyte>(120, -10);
        aCompSbyte[9] = Vector2<byte>(byte(sbA == sbB));

        sbB            = Vector2<sbyte>(10, -120);
        aCompSbyte[10] = Vector2<byte>(byte(sbA == sbB));

        // <=
        sbA            = Vector2<sbyte>(120, -120);
        sbB            = Vector2<sbyte>(120, -120);
        aCompSbyte[11] = Vector2<byte>(byte(sbA <= sbB));

        sbB            = Vector2<sbyte>(110, -120);
        aCompSbyte[12] = Vector2<byte>(byte(sbA <= sbB));

        sbB            = Vector2<sbyte>(121, 0);
        aCompSbyte[13] = Vector2<byte>(byte(sbA <= sbB));

        sbB            = Vector2<sbyte>(120, -120);
        aCompSbyte[14] = Vector2<byte>(byte(sbA <= sbB));

        sbB            = Vector2<sbyte>(120, -120);
        aCompSbyte[15] = Vector2<byte>(byte(sbA <= sbB));

        // <
        sbA            = Vector2<sbyte>(120, -120);
        sbB            = Vector2<sbyte>(120, -120);
        aCompSbyte[16] = Vector2<byte>(byte(sbA < sbB));

        sbB            = Vector2<sbyte>(110, -110);
        aCompSbyte[17] = Vector2<byte>(byte(sbA < sbB));

        sbB            = Vector2<sbyte>(121, 0);
        aCompSbyte[18] = Vector2<byte>(byte(sbA < sbB));

        sbB            = Vector2<sbyte>(121, -121);
        aCompSbyte[19] = Vector2<byte>(byte(sbA < sbB));

        sbB            = Vector2<sbyte>(121, 0);
        aCompSbyte[20] = Vector2<byte>(byte(sbA < sbB));

        // >=
        sbA            = Vector2<sbyte>(120, -120);
        sbB            = Vector2<sbyte>(120, -120);
        aCompSbyte[21] = Vector2<byte>(byte(sbA >= sbB));

        sbB            = Vector2<sbyte>(110, -121);
        aCompSbyte[22] = Vector2<byte>(byte(sbA >= sbB));

        sbB            = Vector2<sbyte>(110, 0);
        aCompSbyte[23] = Vector2<byte>(byte(sbA >= sbB));

        sbB            = Vector2<sbyte>(110, -120);
        aCompSbyte[24] = Vector2<byte>(byte(sbA >= sbB));

        sbB            = Vector2<sbyte>(127, -120);
        aCompSbyte[25] = Vector2<byte>(byte(sbA >= sbB));

        // >
        sbA            = Vector2<sbyte>(120, -120);
        sbB            = Vector2<sbyte>(120, -120);
        aCompSbyte[26] = Vector2<byte>(byte(sbA > sbB));

        sbB            = Vector2<sbyte>(110, -121);
        aCompSbyte[27] = Vector2<byte>(byte(sbA > sbB));

        sbB            = Vector2<sbyte>(121, 0);
        aCompSbyte[28] = Vector2<byte>(byte(sbA > sbB));

        sbB            = Vector2<sbyte>(110, -119);
        aCompSbyte[29] = Vector2<byte>(byte(sbA > sbB));

        sbB            = Vector2<sbyte>(127, -121);
        aCompSbyte[30] = Vector2<byte>(byte(sbA > sbB));

        // !=
        sbA            = Vector2<sbyte>(120, -120);
        sbB            = Vector2<sbyte>(120, -120);
        aCompSbyte[31] = Vector2<byte>(byte(sbA != sbB));

        sbB            = Vector2<sbyte>(110, -120);
        aCompSbyte[32] = Vector2<byte>(byte(sbA != sbB));

        sbB            = Vector2<sbyte>(120, 0);
        aCompSbyte[33] = Vector2<byte>(byte(sbA != sbB));

        sbB            = Vector2<sbyte>(12, -120);
        aCompSbyte[34] = Vector2<byte>(byte(sbA != sbB));

        sbB            = Vector2<sbyte>(120, 0);
        aCompSbyte[35] = Vector2<byte>(byte(sbA != sbB));

        sbB            = Vector2<sbyte>(0, 0);
        aCompSbyte[36] = Vector2<byte>(byte(sbA != sbB));
    }

    // byte
    if (blockIdx.x == 1)
    {
        Vector2<byte> bA(12, 120);
        Vector2<byte> bB(120, 180);
        Vector2<byte> bC(12, 20);

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
        // aDataByte[9]  = Vector2<byte>::Exp(bC);
        // aDataByte[10] = Vector2<byte>::Ln(bC);
        // aDataByte[11] = Vector2<byte>::Sqrt(bC);
        // aDataByte[12] = Vector2<byte>::Abs(bA);
        aDataByte[13] = Vector2<byte>::AbsDiff(bA, bB);

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

        aDataByte[19] = Vector2<byte>::Min(bA, bB);
        aDataByte[20] = Vector2<byte>::Max(bA, bB);

        aDataByte[21] = bA;
        aDataByte[21].Min(bB);
        aDataByte[22] = bA;
        aDataByte[22].Max(bB);

        aCompByte[0] = Vector2<byte>::CompareEQ(bA, bC);
        aCompByte[1] = Vector2<byte>::CompareLE(bA, bC);
        aCompByte[2] = Vector2<byte>::CompareLT(bA, bC);
        aCompByte[3] = Vector2<byte>::CompareGE(bA, bC);
        aCompByte[4] = Vector2<byte>::CompareGT(bA, bC);
        aCompByte[5] = Vector2<byte>::CompareNEQ(bA, bC);

        // ==
        bA           = Vector2<byte>(120, 200);
        bB           = Vector2<byte>(120, 200);
        aCompByte[6] = Vector2<byte>(byte(bA == bB));

        bB           = Vector2<byte>(0, 200);
        aCompByte[7] = Vector2<byte>(byte(bA == bB));

        bB           = Vector2<byte>(120, 0);
        aCompByte[8] = Vector2<byte>(byte(bA == bB));

        bB           = Vector2<byte>(120, 100);
        aCompByte[9] = Vector2<byte>(byte(bA == bB));

        bB            = Vector2<byte>(110, 200);
        aCompByte[10] = Vector2<byte>(byte(bA == bB));

        // <=
        bA            = Vector2<byte>(120, 200);
        bB            = Vector2<byte>(120, 200);
        aCompByte[11] = Vector2<byte>(byte(bA <= bB));

        bB            = Vector2<byte>(110, 200);
        aCompByte[12] = Vector2<byte>(byte(bA <= bB));

        bB            = Vector2<byte>(121, 220);
        aCompByte[13] = Vector2<byte>(byte(bA <= bB));

        bB            = Vector2<byte>(120, 200);
        aCompByte[14] = Vector2<byte>(byte(bA <= bB));

        bB            = Vector2<byte>(120, 200);
        aCompByte[15] = Vector2<byte>(byte(bA <= bB));

        // <
        bA            = Vector2<byte>(120, 200);
        bB            = Vector2<byte>(120, 200);
        aCompByte[16] = Vector2<byte>(byte(bA < bB));

        bB            = Vector2<byte>(110, 220);
        aCompByte[17] = Vector2<byte>(byte(bA < bB));

        bB            = Vector2<byte>(121, 0);
        aCompByte[18] = Vector2<byte>(byte(bA < bB));

        bB            = Vector2<byte>(121, 220);
        aCompByte[19] = Vector2<byte>(byte(bA < bB));

        bB            = Vector2<byte>(121, 220);
        aCompByte[20] = Vector2<byte>(byte(bA < bB));

        // >=
        bA            = Vector2<byte>(120, 200);
        bB            = Vector2<byte>(120, 200);
        aCompByte[21] = Vector2<byte>(byte(bA >= bB));

        bB            = Vector2<byte>(110, 0);
        aCompByte[22] = Vector2<byte>(byte(bA >= bB));

        bB            = Vector2<byte>(110, 220);
        aCompByte[23] = Vector2<byte>(byte(bA >= bB));

        bB            = Vector2<byte>(110, 200);
        aCompByte[24] = Vector2<byte>(byte(bA >= bB));

        bB            = Vector2<byte>(127, 200);
        aCompByte[25] = Vector2<byte>(byte(bA >= bB));

        // >
        bA            = Vector2<byte>(120, 200);
        bB            = Vector2<byte>(120, 200);
        aCompByte[26] = Vector2<byte>(byte(bA > bB));

        bB            = Vector2<byte>(110, 0);
        aCompByte[27] = Vector2<byte>(byte(bA > bB));

        bB            = Vector2<byte>(121, 220);
        aCompByte[28] = Vector2<byte>(byte(bA > bB));

        bB            = Vector2<byte>(121, 0);
        aCompByte[29] = Vector2<byte>(byte(bA > bB));

        bB            = Vector2<byte>(110, 220);
        aCompByte[30] = Vector2<byte>(byte(bA > bB));

        // !=
        bA            = Vector2<byte>(120, 200);
        bB            = Vector2<byte>(120, 200);
        aCompByte[31] = Vector2<byte>(byte(bA != bB));

        bB            = Vector2<byte>(110, 200);
        aCompByte[32] = Vector2<byte>(byte(bA != bB));

        bB            = Vector2<byte>(120, 220);
        aCompByte[33] = Vector2<byte>(byte(bA != bB));

        bB            = Vector2<byte>(120, 20);
        aCompByte[34] = Vector2<byte>(byte(bA != bB));

        bB            = Vector2<byte>(120, 0);
        aCompByte[35] = Vector2<byte>(byte(bA != bB));

        bB            = Vector2<byte>(0, 0);
        aCompByte[36] = Vector2<byte>(byte(bA != bB));
    }

    // short
    if (blockIdx.x == 2)
    {
        Vector2<short> sA(12, -30000);
        Vector2<short> sB(120, -3000);
        Vector2<short> sC(12, -30000);

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
        // aDataShort[9]  = Vector2<short>::Exp(sC);
        // aDataShort[10] = Vector2<short>::Ln(sC);
        // aDataShort[11] = Vector2<short>::Sqrt(sC);
        aDataShort[12] = Vector2<short>::Abs(sA);
        aDataShort[13] = Vector2<short>::AbsDiff(sA, sB);

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

        aDataShort[19] = Vector2<short>::Min(sA, sB);
        aDataShort[20] = Vector2<short>::Max(sA, sB);

        aDataShort[21] = sA;
        aDataShort[21].Min(sB);
        aDataShort[22] = sA;
        aDataShort[22].Max(sB);

        aCompShort[0] = Vector2<short>::CompareEQ(sA, sC);
        aCompShort[1] = Vector2<short>::CompareLE(sA, sC);
        aCompShort[2] = Vector2<short>::CompareLT(sA, sC);
        aCompShort[3] = Vector2<short>::CompareGE(sA, sC);
        aCompShort[4] = Vector2<short>::CompareGT(sA, sC);
        aCompShort[5] = Vector2<short>::CompareNEQ(sA, sC);

        // ==
        sA            = Vector2<short>(120, -120);
        sB            = Vector2<short>(120, -120);
        aCompShort[6] = Vector2<byte>(byte(sA == sB));

        sB            = Vector2<short>(0, -120);
        aCompShort[7] = Vector2<byte>(byte(sA == sB));

        sB            = Vector2<short>(120, 0);
        aCompShort[8] = Vector2<byte>(byte(sA == sB));

        sB            = Vector2<short>(10, -120);
        aCompShort[9] = Vector2<byte>(byte(sA == sB));

        sB             = Vector2<short>(120, -10);
        aCompShort[10] = Vector2<byte>(byte(sA == sB));

        // <=
        sA             = Vector2<short>(120, -120);
        sB             = Vector2<short>(120, -120);
        aCompShort[11] = Vector2<byte>(byte(sA <= sB));

        sB             = Vector2<short>(110, -120);
        aCompShort[12] = Vector2<byte>(byte(sA <= sB));

        sB             = Vector2<short>(121, 0);
        aCompShort[13] = Vector2<byte>(byte(sA <= sB));

        sB             = Vector2<short>(120, -120);
        aCompShort[14] = Vector2<byte>(byte(sA <= sB));

        sB             = Vector2<short>(120, -120);
        aCompShort[15] = Vector2<byte>(byte(sA <= sB));

        // <
        sA             = Vector2<short>(120, -120);
        sB             = Vector2<short>(120, -120);
        aCompShort[16] = Vector2<byte>(byte(sA < sB));

        sB             = Vector2<short>(110, -110);
        aCompShort[17] = Vector2<byte>(byte(sA < sB));

        sB             = Vector2<short>(121, 0);
        aCompShort[18] = Vector2<byte>(byte(sA < sB));

        sB             = Vector2<short>(121, -121);
        aCompShort[19] = Vector2<byte>(byte(sA < sB));

        sB             = Vector2<short>(121, 0);
        aCompShort[20] = Vector2<byte>(byte(sA < sB));

        // >=
        sA             = Vector2<short>(120, -120);
        sB             = Vector2<short>(120, -120);
        aCompShort[21] = Vector2<byte>(byte(sA >= sB));

        sB             = Vector2<short>(110, -121);
        aCompShort[22] = Vector2<byte>(byte(sA >= sB));

        sB             = Vector2<short>(110, 0);
        aCompShort[23] = Vector2<byte>(byte(sA >= sB));

        sB             = Vector2<short>(110, -120);
        aCompShort[24] = Vector2<byte>(byte(sA >= sB));

        sB             = Vector2<short>(127, -120);
        aCompShort[25] = Vector2<byte>(byte(sA >= sB));

        // >
        sA             = Vector2<short>(120, -120);
        sB             = Vector2<short>(120, -120);
        aCompShort[26] = Vector2<byte>(byte(sA > sB));

        sB             = Vector2<short>(110, -121);
        aCompShort[27] = Vector2<byte>(byte(sA > sB));

        sB             = Vector2<short>(121, 0);
        aCompShort[28] = Vector2<byte>(byte(sA > sB));

        sB             = Vector2<short>(121, -121);
        aCompShort[29] = Vector2<byte>(byte(sA > sB));

        sB             = Vector2<short>(119, 121);
        aCompShort[30] = Vector2<byte>(byte(sA > sB));

        // !=
        sA             = Vector2<short>(120, -120);
        sB             = Vector2<short>(120, -120);
        aCompShort[31] = Vector2<byte>(byte(sA != sB));

        sB             = Vector2<short>(110, -120);
        aCompShort[32] = Vector2<byte>(byte(sA != sB));

        sB             = Vector2<short>(120, 0);
        aCompShort[33] = Vector2<byte>(byte(sA != sB));

        sB             = Vector2<short>(120, -20);
        aCompShort[34] = Vector2<byte>(byte(sA != sB));

        sB             = Vector2<short>(120, 0);
        aCompShort[35] = Vector2<byte>(byte(sA != sB));

        sB             = Vector2<short>(0, 0);
        aCompShort[36] = Vector2<byte>(byte(sA != sB));
    }

    // ushort
    if (blockIdx.x == 3)
    {
        Vector2<ushort> usA(12, 60000);
        Vector2<ushort> usB(120, 7000);
        Vector2<ushort> usC(12, 20);

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
        // aDataUShort[9]  = Vector2<ushort>::Exp(usC);
        // aDataUShort[10] = Vector2<ushort>::Ln(usC);
        // aDataUShort[11] = Vector2<ushort>::Sqrt(usC);
        // aDataUShort[12] = Vector2<ushort>::Abs(usA);
        aDataUShort[13] = Vector2<ushort>::AbsDiff(usA, usB);

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

        aDataUShort[19] = Vector2<ushort>::Min(usA, usB);
        aDataUShort[20] = Vector2<ushort>::Max(usA, usB);

        aDataUShort[21] = usA;
        aDataUShort[21].Min(usB);
        aDataUShort[22] = usA;
        aDataUShort[22].Max(usB);

        aCompUShort[0] = Vector2<ushort>::CompareEQ(usA, usC);
        aCompUShort[1] = Vector2<ushort>::CompareLE(usA, usC);
        aCompUShort[2] = Vector2<ushort>::CompareLT(usA, usC);
        aCompUShort[3] = Vector2<ushort>::CompareGE(usA, usC);
        aCompUShort[4] = Vector2<ushort>::CompareGT(usA, usC);
        aCompUShort[5] = Vector2<ushort>::CompareNEQ(usA, usC);

        // ==
        usA            = Vector2<ushort>(120, 200);
        usB            = Vector2<ushort>(120, 200);
        aCompUShort[6] = Vector2<byte>(byte(usA == usB));

        usB            = Vector2<ushort>(0, 200);
        aCompUShort[7] = Vector2<byte>(byte(usA == usB));

        usB            = Vector2<ushort>(120, 0);
        aCompUShort[8] = Vector2<byte>(byte(usA == usB));

        usB            = Vector2<ushort>(10, 200);
        aCompUShort[9] = Vector2<byte>(byte(usA == usB));

        usB             = Vector2<ushort>(120, 20);
        aCompUShort[10] = Vector2<byte>(byte(usA == usB));

        // <=
        usA             = Vector2<ushort>(120, 200);
        usB             = Vector2<ushort>(120, 200);
        aCompUShort[11] = Vector2<byte>(byte(usA <= usB));

        usB             = Vector2<ushort>(110, 200);
        aCompUShort[12] = Vector2<byte>(byte(usA <= usB));

        usB             = Vector2<ushort>(121, 220);
        aCompUShort[13] = Vector2<byte>(byte(usA <= usB));

        usB             = Vector2<ushort>(120, 200);
        aCompUShort[14] = Vector2<byte>(byte(usA <= usB));

        usB             = Vector2<ushort>(120, 200);
        aCompUShort[15] = Vector2<byte>(byte(usA <= usB));

        // <
        usA             = Vector2<ushort>(120, 200);
        usB             = Vector2<ushort>(120, 200);
        aCompUShort[16] = Vector2<byte>(byte(usA < usB));

        usB             = Vector2<ushort>(110, 220);
        aCompUShort[17] = Vector2<byte>(byte(usA < usB));

        usB             = Vector2<ushort>(121, 0);
        aCompUShort[18] = Vector2<byte>(byte(usA < usB));

        usB             = Vector2<ushort>(121, 220);
        aCompUShort[19] = Vector2<byte>(byte(usA < usB));

        usB             = Vector2<ushort>(121, 220);
        aCompUShort[20] = Vector2<byte>(byte(usA < usB));

        // >=
        usA             = Vector2<ushort>(120, 200);
        usB             = Vector2<ushort>(120, 200);
        aCompUShort[21] = Vector2<byte>(byte(usA >= usB));

        usB             = Vector2<ushort>(110, 0);
        aCompUShort[22] = Vector2<byte>(byte(usA >= usB));

        usB             = Vector2<ushort>(110, 220);
        aCompUShort[23] = Vector2<byte>(byte(usA >= usB));

        usB             = Vector2<ushort>(110, 200);
        aCompUShort[24] = Vector2<byte>(byte(usA >= usB));

        usB             = Vector2<ushort>(121, 200);
        aCompUShort[25] = Vector2<byte>(byte(usA >= usB));

        // >
        usA             = Vector2<ushort>(120, 200);
        usB             = Vector2<ushort>(120, 200);
        aCompUShort[26] = Vector2<byte>(byte(usA > usB));

        usB             = Vector2<ushort>(110, 0);
        aCompUShort[27] = Vector2<byte>(byte(usA > usB));

        usB             = Vector2<ushort>(121, 220);
        aCompUShort[28] = Vector2<byte>(byte(usA > usB));

        usB             = Vector2<ushort>(121, 0);
        aCompUShort[29] = Vector2<byte>(byte(usA > usB));

        usB             = Vector2<ushort>(110, 220);
        aCompUShort[30] = Vector2<byte>(byte(usA > usB));

        // !=
        usA             = Vector2<ushort>(120, 200);
        usB             = Vector2<ushort>(120, 200);
        aCompUShort[31] = Vector2<byte>(byte(usA != usB));

        usB             = Vector2<ushort>(110, 200);
        aCompUShort[32] = Vector2<byte>(byte(usA != usB));

        usB             = Vector2<ushort>(120, 220);
        aCompUShort[33] = Vector2<byte>(byte(usA != usB));

        usB             = Vector2<ushort>(10, 200);
        aCompUShort[34] = Vector2<byte>(byte(usA != usB));

        usB             = Vector2<ushort>(120, 0);
        aCompUShort[35] = Vector2<byte>(byte(usA != usB));

        usB             = Vector2<ushort>(0, 0);
        aCompUShort[36] = Vector2<byte>(byte(usA != usB));
    }

    // BFloat
    if (blockIdx.x == 4)
    {
        Vector2<BFloat16> bfA(BFloat16(12.4f), BFloat16(-30000.2f));
        Vector2<BFloat16> bfB(BFloat16(120.1f), BFloat16(-3000.1f));
        Vector2<BFloat16> bfC(BFloat16(12.4f), BFloat16(-30000.2f));

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
        aDataBFloat[9]  = Vector2<BFloat16>::Exp(bfC);
        aDataBFloat[10] = Vector2<BFloat16>::Ln(bfC);
        aDataBFloat[11] = Vector2<BFloat16>::Sqrt(bfC);
        aDataBFloat[12] = Vector2<BFloat16>::Abs(bfA);
        aDataBFloat[13] = Vector2<BFloat16>::AbsDiff(bfA, bfB);

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

        aDataBFloat[19] = Vector2<BFloat16>::Min(bfA, bfB);
        aDataBFloat[20] = Vector2<BFloat16>::Max(bfA, bfB);

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

        aDataBFloat[28] = Vector2<BFloat16>::Round(bfA);
        aDataBFloat[29] = Vector2<BFloat16>::Ceil(bfA);
        aDataBFloat[30] = Vector2<BFloat16>::Floor(bfA);
        aDataBFloat[31] = Vector2<BFloat16>::RoundNearest(bfA);
        aDataBFloat[32] = Vector2<BFloat16>::RoundZero(bfA);

        Vector2<float> bvec4(1.0f, 2.0f);
        aDataBFloat[33] = Vector2<BFloat16>(bvec4);
        Vector2<float> bvec42(aDataBFloat[33]);
        aDataBFloat[34] = Vector2<BFloat16>(bvec4 + bvec42);

        aCompBFloat[0] = Vector2<BFloat16>::CompareEQ(bfA, bfC);
        aCompBFloat[1] = Vector2<BFloat16>::CompareLE(bfA, bfC);
        aCompBFloat[2] = Vector2<BFloat16>::CompareLT(bfA, bfC);
        aCompBFloat[3] = Vector2<BFloat16>::CompareGE(bfA, bfC);
        aCompBFloat[4] = Vector2<BFloat16>::CompareGT(bfA, bfC);
        aCompBFloat[5] = Vector2<BFloat16>::CompareNEQ(bfA, bfC);

        // ==
        bfA            = Vector2<BFloat16>(BFloat16(120), BFloat16(-120));
        bfB            = Vector2<BFloat16>(BFloat16(120), BFloat16(-120));
        aCompBFloat[6] = Vector2<byte>(byte(bfA == bfB));

        bfB            = Vector2<BFloat16>(BFloat16(0), BFloat16(-120));
        aCompBFloat[7] = Vector2<byte>(byte(bfA == bfB));

        bfB            = Vector2<BFloat16>(BFloat16(120), BFloat16(0));
        aCompBFloat[8] = Vector2<byte>(byte(bfA == bfB));

        bfB            = Vector2<BFloat16>(BFloat16(10), BFloat16(-120));
        aCompBFloat[9] = Vector2<byte>(byte(bfA == bfB));

        bfB             = Vector2<BFloat16>(BFloat16(120), BFloat16(-10));
        aCompBFloat[10] = Vector2<byte>(byte(bfA == bfB));

        // <=
        bfA             = Vector2<BFloat16>(BFloat16(120), BFloat16(-120));
        bfB             = Vector2<BFloat16>(BFloat16(120), BFloat16(-120));
        aCompBFloat[11] = Vector2<byte>(byte(bfA <= bfB));

        bfB             = Vector2<BFloat16>(BFloat16(110), BFloat16(-120));
        aCompBFloat[12] = Vector2<byte>(byte(bfA <= bfB));

        bfB             = Vector2<BFloat16>(BFloat16(121), BFloat16(0));
        aCompBFloat[13] = Vector2<byte>(byte(bfA <= bfB));

        bfB             = Vector2<BFloat16>(BFloat16(120), BFloat16(-120));
        aCompBFloat[14] = Vector2<byte>(byte(bfA <= bfB));

        bfB             = Vector2<BFloat16>(BFloat16(120), BFloat16(-120));
        aCompBFloat[15] = Vector2<byte>(byte(bfA <= bfB));

        // <
        bfA             = Vector2<BFloat16>(BFloat16(120), BFloat16(-120));
        bfB             = Vector2<BFloat16>(BFloat16(120), BFloat16(-120));
        aCompBFloat[16] = Vector2<byte>(byte(bfA < bfB));

        bfB             = Vector2<BFloat16>(BFloat16(110), BFloat16(-110));
        aCompBFloat[17] = Vector2<byte>(byte(bfA < bfB));

        bfB             = Vector2<BFloat16>(BFloat16(121), BFloat16(0));
        aCompBFloat[18] = Vector2<byte>(byte(bfA < bfB));

        bfB             = Vector2<BFloat16>(BFloat16(121), BFloat16(-121));
        aCompBFloat[19] = Vector2<byte>(byte(bfA < bfB));

        bfB             = Vector2<BFloat16>(BFloat16(121), BFloat16(0));
        aCompBFloat[20] = Vector2<byte>(byte(bfA < bfB));

        // >=
        bfA             = Vector2<BFloat16>(BFloat16(120), BFloat16(-120));
        bfB             = Vector2<BFloat16>(BFloat16(120), BFloat16(-120));
        aCompBFloat[21] = Vector2<byte>(byte(bfA >= bfB));

        bfB             = Vector2<BFloat16>(BFloat16(110), BFloat16(-121));
        aCompBFloat[22] = Vector2<byte>(byte(bfA >= bfB));

        bfB             = Vector2<BFloat16>(BFloat16(110), BFloat16(0));
        aCompBFloat[23] = Vector2<byte>(byte(bfA >= bfB));

        bfB             = Vector2<BFloat16>(BFloat16(110), BFloat16(-120));
        aCompBFloat[24] = Vector2<byte>(byte(bfA >= bfB));

        bfB             = Vector2<BFloat16>(BFloat16(121), BFloat16(-120));
        aCompBFloat[25] = Vector2<byte>(byte(bfA >= bfB));

        // >
        bfA             = Vector2<BFloat16>(BFloat16(120), BFloat16(-120));
        bfB             = Vector2<BFloat16>(BFloat16(120), BFloat16(-120));
        aCompBFloat[26] = Vector2<byte>(byte(bfA > bfB));

        bfB             = Vector2<BFloat16>(BFloat16(110), BFloat16(-121));
        aCompBFloat[27] = Vector2<byte>(byte(bfA > bfB));

        bfB             = Vector2<BFloat16>(BFloat16(121), BFloat16(0));
        aCompBFloat[28] = Vector2<byte>(byte(bfA > bfB));

        bfB             = Vector2<BFloat16>(BFloat16(121), BFloat16(-121));
        aCompBFloat[29] = Vector2<byte>(byte(bfA > bfB));

        bfB             = Vector2<BFloat16>(BFloat16(110), BFloat16(0));
        aCompBFloat[30] = Vector2<byte>(byte(bfA > bfB));

        // !=
        bfA             = Vector2<BFloat16>(BFloat16(120), BFloat16(-120));
        bfB             = Vector2<BFloat16>(BFloat16(120), BFloat16(-120));
        aCompBFloat[31] = Vector2<byte>(byte(bfA != bfB));

        bfB             = Vector2<BFloat16>(BFloat16(110), BFloat16(-120));
        aCompBFloat[32] = Vector2<byte>(byte(bfA != bfB));

        bfB             = Vector2<BFloat16>(BFloat16(120), BFloat16(0));
        aCompBFloat[33] = Vector2<byte>(byte(bfA != bfB));

        bfB             = Vector2<BFloat16>(BFloat16(10), BFloat16(-120));
        aCompBFloat[34] = Vector2<byte>(byte(bfA != bfB));

        bfB             = Vector2<BFloat16>(BFloat16(120), BFloat16(0));
        aCompBFloat[35] = Vector2<byte>(byte(bfA != bfB));

        bfB             = Vector2<BFloat16>(BFloat16(0), BFloat16(0));
        aCompBFloat[36] = Vector2<byte>(byte(bfA != bfB));
    }

    // Half
    if (blockIdx.x == 5)
    {
        Vector2<HalfFp16> hA(HalfFp16(12.4f), HalfFp16(-30000.2f));
        Vector2<HalfFp16> hB(HalfFp16(120.1f), HalfFp16(-3000.1f));
        Vector2<HalfFp16> hC(HalfFp16(12.4f), HalfFp16(-30000.2f));

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
        aDataHalf[9]  = Vector2<HalfFp16>::Exp(hC);
        aDataHalf[10] = Vector2<HalfFp16>::Ln(hC);
        aDataHalf[11] = Vector2<HalfFp16>::Sqrt(hC);
        aDataHalf[12] = Vector2<HalfFp16>::Abs(hA);
        aDataHalf[13] = Vector2<HalfFp16>::AbsDiff(hA, hB);

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

        aDataHalf[19] = Vector2<HalfFp16>::Min(hA, hB);
        aDataHalf[20] = Vector2<HalfFp16>::Max(hA, hB);

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

        aDataHalf[28] = Vector2<HalfFp16>::Round(hA);
        aDataHalf[29] = Vector2<HalfFp16>::Ceil(hA);
        aDataHalf[30] = Vector2<HalfFp16>::Floor(hA);
        aDataHalf[31] = Vector2<HalfFp16>::RoundNearest(hA);
        aDataHalf[32] = Vector2<HalfFp16>::RoundZero(hA);

        Vector2<float> hvec4(1.0f, 2.0f);
        aDataHalf[33] = Vector2<HalfFp16>(hvec4);
        Vector2<float> hvec42(aDataHalf[33]);
        aDataHalf[34] = Vector2<HalfFp16>(hvec4 + hvec42);

        aCompHalf[0] = Vector2<HalfFp16>::CompareEQ(hA, hC);
        aCompHalf[1] = Vector2<HalfFp16>::CompareLE(hA, hC);
        aCompHalf[2] = Vector2<HalfFp16>::CompareLT(hA, hC);
        aCompHalf[3] = Vector2<HalfFp16>::CompareGE(hA, hC);
        aCompHalf[4] = Vector2<HalfFp16>::CompareGT(hA, hC);
        aCompHalf[5] = Vector2<HalfFp16>::CompareNEQ(hA, hC);

        // ==
        hA           = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-120));
        hB           = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-120));
        aCompHalf[6] = Vector2<byte>(byte(hA == hB));

        hB           = Vector2<HalfFp16>(HalfFp16(0), HalfFp16(-120));
        aCompHalf[7] = Vector2<byte>(byte(hA == hB));

        hB           = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(0));
        aCompHalf[8] = Vector2<byte>(byte(hA == hB));

        hB           = Vector2<HalfFp16>(HalfFp16(10), HalfFp16(-120));
        aCompHalf[9] = Vector2<byte>(byte(hA == hB));

        hB            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-10));
        aCompHalf[10] = Vector2<byte>(byte(hA == hB));

        // <=
        hA            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-120));
        hB            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-120));
        aCompHalf[11] = Vector2<byte>(byte(hA <= hB));

        hB            = Vector2<HalfFp16>(HalfFp16(110), HalfFp16(-120));
        aCompHalf[12] = Vector2<byte>(byte(hA <= hB));

        hB            = Vector2<HalfFp16>(HalfFp16(121), HalfFp16(0));
        aCompHalf[13] = Vector2<byte>(byte(hA <= hB));

        hB            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-120));
        aCompHalf[14] = Vector2<byte>(byte(hA <= hB));

        hB            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-120));
        aCompHalf[15] = Vector2<byte>(byte(hA <= hB));

        // <
        hA            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-120));
        hB            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-120));
        aCompHalf[16] = Vector2<byte>(byte(hA < hB));

        hB            = Vector2<HalfFp16>(HalfFp16(110), HalfFp16(-110));
        aCompHalf[17] = Vector2<byte>(byte(hA < hB));

        hB            = Vector2<HalfFp16>(HalfFp16(121), HalfFp16(0));
        aCompHalf[18] = Vector2<byte>(byte(hA < hB));

        hB            = Vector2<HalfFp16>(HalfFp16(121), HalfFp16(-121));
        aCompHalf[19] = Vector2<byte>(byte(hA < hB));

        hB            = Vector2<HalfFp16>(HalfFp16(121), HalfFp16(0));
        aCompHalf[20] = Vector2<byte>(byte(hA < hB));

        // >=
        hA            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-120));
        hB            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-120));
        aCompHalf[21] = Vector2<byte>(byte(hA >= hB));

        hB            = Vector2<HalfFp16>(HalfFp16(110), HalfFp16(-121));
        aCompHalf[22] = Vector2<byte>(byte(hA >= hB));

        hB            = Vector2<HalfFp16>(HalfFp16(110), HalfFp16(0));
        aCompHalf[23] = Vector2<byte>(byte(hA >= hB));

        hB            = Vector2<HalfFp16>(HalfFp16(110), HalfFp16(-120));
        aCompHalf[24] = Vector2<byte>(byte(hA >= hB));

        hB            = Vector2<HalfFp16>(HalfFp16(127), HalfFp16(-120));
        aCompHalf[25] = Vector2<byte>(byte(hA >= hB));

        // >
        hA            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-120));
        hB            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-120));
        aCompHalf[26] = Vector2<byte>(byte(hA > hB));

        hB            = Vector2<HalfFp16>(HalfFp16(110), HalfFp16(-121));
        aCompHalf[27] = Vector2<byte>(byte(hA > hB));

        hB            = Vector2<HalfFp16>(HalfFp16(121), HalfFp16(0));
        aCompHalf[28] = Vector2<byte>(byte(hA > hB));

        hB            = Vector2<HalfFp16>(HalfFp16(121), HalfFp16(-121));
        aCompHalf[29] = Vector2<byte>(byte(hA > hB));

        hB            = Vector2<HalfFp16>(HalfFp16(110), HalfFp16(-119));
        aCompHalf[30] = Vector2<byte>(byte(hA > hB));

        // !=
        hA            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-120));
        hB            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(-120));
        aCompHalf[31] = Vector2<byte>(byte(hA != hB));

        hB            = Vector2<HalfFp16>(HalfFp16(110), HalfFp16(-120));
        aCompHalf[32] = Vector2<byte>(byte(hA != hB));

        hB            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(0));
        aCompHalf[33] = Vector2<byte>(byte(hA != hB));

        hB            = Vector2<HalfFp16>(HalfFp16(10), HalfFp16(-120));
        aCompHalf[34] = Vector2<byte>(byte(hA != hB));

        hB            = Vector2<HalfFp16>(HalfFp16(120), HalfFp16(0));
        aCompHalf[35] = Vector2<byte>(byte(hA != hB));

        hB            = Vector2<HalfFp16>(HalfFp16(0), HalfFp16(0));
        aCompHalf[36] = Vector2<byte>(byte(hA != hB));
    }

    // Float
    if (blockIdx.x == 6)
    {
        Vector2<float> fA(12.4f, -30000.2f);
        Vector2<float> fB(120.1f, -3000.1f);
        Vector2<float> fC(12.4f, -30000.2f);

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
        aDataFloat[9]  = Vector2<float>::Exp(fC);
        aDataFloat[10] = Vector2<float>::Ln(fC);
        aDataFloat[11] = Vector2<float>::Sqrt(fC);
        aDataFloat[12] = Vector2<float>::Abs(fA);
        aDataFloat[13] = Vector2<float>::AbsDiff(fA, fB);

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

        aDataFloat[19] = Vector2<float>::Min(fA, fB);
        aDataFloat[20] = Vector2<float>::Max(fA, fB);

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

        aDataFloat[28] = Vector2<float>::Round(fA);
        aDataFloat[29] = Vector2<float>::Ceil(fA);
        aDataFloat[30] = Vector2<float>::Floor(fA);
        aDataFloat[31] = Vector2<float>::RoundNearest(fA);
        aDataFloat[32] = Vector2<float>::RoundZero(fA);

        Vector2<float> bvec4(1.0f, 2.0f);
        aDataFloat[33] = Vector2<HalfFp16>(bvec4);
        Vector2<float> bvec42(aDataFloat[33]);
        aDataFloat[34] = Vector2<HalfFp16>(bvec4 + bvec42);

        aCompFloat[0] = Vector2<float>::CompareEQ(fA, fC);
        aCompFloat[1] = Vector2<float>::CompareLE(fA, fC);
        aCompFloat[2] = Vector2<float>::CompareLT(fA, fC);
        aCompFloat[3] = Vector2<float>::CompareGE(fA, fC);
        aCompFloat[4] = Vector2<float>::CompareGT(fA, fC);
        aCompFloat[5] = Vector2<float>::CompareNEQ(fA, fC);

        // ==
        fA            = Vector2<float>(120, -120);
        fB            = Vector2<float>(120, -120);
        aCompFloat[6] = Vector2<byte>(byte(fA == fB));

        fB            = Vector2<float>(0, -120);
        aCompFloat[7] = Vector2<byte>(byte(fA == fB));

        fB            = Vector2<float>(120, 0);
        aCompFloat[8] = Vector2<byte>(byte(fA == fB));

        fB            = Vector2<float>(10, -120);
        aCompFloat[9] = Vector2<byte>(byte(fA == fB));

        fB             = Vector2<float>(120, -10);
        aCompFloat[10] = Vector2<byte>(byte(fA == fB));

        // <=
        fA             = Vector2<float>(120, -120);
        fB             = Vector2<float>(120, -120);
        aCompFloat[11] = Vector2<byte>(byte(fA <= fB));

        fB             = Vector2<float>(110, -120);
        aCompFloat[12] = Vector2<byte>(byte(fA <= fB));

        fB             = Vector2<float>(121, 0);
        aCompFloat[13] = Vector2<byte>(byte(fA <= fB));

        fB             = Vector2<float>(120, -120);
        aCompFloat[14] = Vector2<byte>(byte(fA <= fB));

        fB             = Vector2<float>(120, -120);
        aCompFloat[15] = Vector2<byte>(byte(fA <= fB));

        // <
        fA             = Vector2<float>(120, -120);
        fB             = Vector2<float>(120, -120);
        aCompFloat[16] = Vector2<byte>(byte(fA < fB));

        fB             = Vector2<float>(110, -110);
        aCompFloat[17] = Vector2<byte>(byte(fA < fB));

        fB             = Vector2<float>(121, 0);
        aCompFloat[18] = Vector2<byte>(byte(fA < fB));

        fB             = Vector2<float>(121, -121);
        aCompFloat[19] = Vector2<byte>(byte(fA < fB));

        fB             = Vector2<float>(121, 0);
        aCompFloat[20] = Vector2<byte>(byte(fA < fB));

        // >=
        fA             = Vector2<float>(120, -120);
        fB             = Vector2<float>(120, -120);
        aCompFloat[21] = Vector2<byte>(byte(fA >= fB));

        fB             = Vector2<float>(110, -121);
        aCompFloat[22] = Vector2<byte>(byte(fA >= fB));

        fB             = Vector2<float>(110, 0);
        aCompFloat[23] = Vector2<byte>(byte(fA >= fB));

        fB             = Vector2<float>(110, -120);
        aCompFloat[24] = Vector2<byte>(byte(fA >= fB));

        fB             = Vector2<float>(120, -10);
        aCompFloat[25] = Vector2<byte>(byte(fA >= fB));

        // >
        fA             = Vector2<float>(120, -120);
        fB             = Vector2<float>(120, -120);
        aCompFloat[26] = Vector2<byte>(byte(fA > fB));

        fB             = Vector2<float>(110, -121);
        aCompFloat[27] = Vector2<byte>(byte(fA > fB));

        fB             = Vector2<float>(121, 0);
        aCompFloat[28] = Vector2<byte>(byte(fA > fB));

        fB             = Vector2<float>(121, -121);
        aCompFloat[29] = Vector2<byte>(byte(fA > fB));

        fB             = Vector2<float>(110, -119);
        aCompFloat[30] = Vector2<byte>(byte(fA > fB));

        // !=
        fA             = Vector2<float>(120, -120);
        fB             = Vector2<float>(120, -120);
        aCompFloat[31] = Vector2<byte>(byte(fA != fB));

        fB             = Vector2<float>(110, -120);
        aCompFloat[32] = Vector2<byte>(byte(fA != fB));

        fB             = Vector2<float>(120, 0);
        aCompFloat[33] = Vector2<byte>(byte(fA != fB));

        fB             = Vector2<float>(10, -120);
        aCompFloat[34] = Vector2<byte>(byte(fA != fB));

        fB             = Vector2<float>(120, 0);
        aCompFloat[35] = Vector2<byte>(byte(fA != fB));

        fB             = Vector2<float>(0, 0);
        aCompFloat[36] = Vector2<byte>(byte(fA != fB));
    }
}

void runtest_vector2_kernel(Vector2<byte> *aDataByte, Vector2<byte> *aCompByte,         //
                            Vector2<sbyte> *aDataSByte, Vector2<byte> *aCompSbyte,      //
                            Vector2<short> *aDataShort, Vector2<byte> *aCompShort,      //
                            Vector2<ushort> *aDataUShort, Vector2<byte> *aCompUShort,   //
                            Vector2<BFloat16> *aDataBFloat, Vector2<byte> *aCompBFloat, //
                            Vector2<HalfFp16> *aDataHalf, Vector2<byte> *aCompHalf,     //
                            Vector2<float> *aDataFloat, Vector2<byte> *aCompFloat)
{
    test_vector2_kernel<<<7, 1>>>(aDataByte, aCompByte,     //
                                  aDataSByte, aCompSbyte,   //
                                  aDataShort, aCompShort,   //
                                  aDataUShort, aCompUShort, //
                                  aDataBFloat, aCompBFloat, //
                                  aDataHalf, aCompHalf,     //
                                  aDataFloat, aCompFloat);
}
} // namespace cuda
} // namespace opp