#include "solve.h"
#include <cmath>
#include <common/safeCast.h>
#include <vector>

bool mpp::image::solve(double *aMatrix, double *aVec, int aN)
{
    constexpr double ZERO = 10e-30;
    double rowMax         = 0;
    double xm             = 0;
    std::vector<int> nRow(to_size_t(aN));
    const int nCols = aN + 1;

    bool OK = true;

    for (int i = 1; i <= aN; i++)
    {
        nRow[to_size_t(i - 1)] = i;
    }

    const int NN = aN - 1;
    int iChg     = 0;
    int i        = 1;

    while ((OK) && (i <= NN))
    {
        int iMax     = nRow[to_size_t(i - 1)];
        rowMax       = std::abs(aMatrix[nCols * (iMax - 1) + i - 1]);
        iMax         = i;
        const int jj = i + 1;
        for (int ip = jj; ip <= aN; ip++)
        {
            const int jp = nRow[to_size_t(ip - 1)];
            if (std::abs(aMatrix[nCols * (jp - 1) + i - 1]) > rowMax)
            {
                rowMax = std::abs(aMatrix[nCols * (jp - 1) + i - 1]);
                iMax   = ip;
            }
        }

        if (rowMax <= ZERO)
        {
            OK = false;
        }
        else
        {
            if (nRow[to_size_t(i - 1)] != nRow[to_size_t(iMax - 1)])
            {
                iChg                      = iChg + 1;
                const int nTemp           = nRow[to_size_t(i - 1)];
                nRow[to_size_t(i - 1)]    = nRow[to_size_t(iMax - 1)];
                nRow[to_size_t(iMax - 1)] = nTemp;
            }
            const int i1 = nRow[to_size_t(i - 1)];

            // Gaussian elimination step
            for (int j = jj; j <= aN; j++)
            {
                const int j1 = nRow[to_size_t(j - 1)];

                xm = aMatrix[nCols * (j1 - 1) + i - 1] / aMatrix[nCols * (i1 - 1) + i - 1];

                for (int k = jj; k <= nCols; k++)
                {
                    aMatrix[nCols * (j1 - 1) + k - 1] =
                        aMatrix[nCols * (j1 - 1) + k - 1] - xm * aMatrix[nCols * (i1 - 1) + k - 1];
                }

                aMatrix[nCols * (j1 - 1) + i - 1] = 0.0;
            }
        }
        i++;
    }

    if (OK)
    {
        const int n1 = nRow[to_size_t(aN - 1)];
        if (std::abs(aMatrix[nCols * (n1 - 1) + aN - 1]) <= ZERO)
        {
            // system has no unique solution
            OK = false;
        }

        else
        {
            // start backward substitution
            aVec[aN - 1] = aMatrix[nCols * (n1 - 1) + nCols - 1] / aMatrix[nCols * (n1 - 1) + aN - 1];

            for (int k = 1; k <= NN; k++)
            {
                i            = NN - k + 1;
                const int jj = i + 1;
                const int n2 = nRow[to_size_t(i - 1)];
                double sum   = 0.0;
                for (int kk = jj; kk <= aN; kk++)
                {
                    sum = sum - aMatrix[nCols * (n2 - 1) + kk - 1] * aVec[kk - 1];
                }
                aVec[i - 1] = (aMatrix[nCols * (n2 - 1) + aN] + sum) / aMatrix[nCols * (n2 - 1) + i - 1];
            }
        }
    }

    return OK;
}

bool mpp::image::solve(float *aMatrix, float *aVec, int aN)
{
    constexpr float ZERO = 10e-15f;
    float rowMax         = 0;
    float xm             = 0;
    std::vector<int> nRow(to_size_t(aN));
    const int nCols = aN + 1;

    bool OK = true;

    for (int i = 1; i <= aN; i++)
    {
        nRow[to_size_t(i - 1)] = i;
    }

    const int NN = aN - 1;
    int iChg     = 0;
    int i        = 1;

    while ((OK) && (i <= NN))
    {
        int iMax     = nRow[to_size_t(i - 1)];
        rowMax       = std::abs(aMatrix[nCols * (iMax - 1) + i - 1]);
        iMax         = i;
        const int jj = i + 1;
        for (int ip = jj; ip <= aN; ip++)
        {
            const int jp = nRow[to_size_t(ip - 1)];
            if (std::abs(aMatrix[nCols * (jp - 1) + i - 1]) > rowMax)
            {
                rowMax = std::abs(aMatrix[nCols * (jp - 1) + i - 1]);
                iMax   = ip;
            }
        }

        if (rowMax <= ZERO)
        {
            OK = false;
        }
        else
        {
            if (nRow[to_size_t(i - 1)] != nRow[to_size_t(iMax - 1)])
            {
                iChg                      = iChg + 1;
                const int nTemp           = nRow[to_size_t(i - 1)];
                nRow[to_size_t(i - 1)]    = nRow[to_size_t(iMax - 1)];
                nRow[to_size_t(iMax - 1)] = nTemp;
            }
            const int i1 = nRow[to_size_t(i - 1)];

            // Gaussian elimination step
            for (int j = jj; j <= aN; j++)
            {
                const int j1 = nRow[to_size_t(j - 1)];

                xm = aMatrix[nCols * (j1 - 1) + i - 1] / aMatrix[nCols * (i1 - 1) + i - 1];

                for (int k = jj; k <= nCols; k++)
                {
                    aMatrix[nCols * (j1 - 1) + k - 1] =
                        aMatrix[nCols * (j1 - 1) + k - 1] - xm * aMatrix[nCols * (i1 - 1) + k - 1];
                }

                aMatrix[nCols * (j1 - 1) + i - 1] = 0.0;
            }
        }
        i++;
    }

    if (OK)
    {
        const int n1 = nRow[to_size_t(aN - 1)];
        if (std::abs(aMatrix[nCols * (n1 - 1) + aN - 1]) <= ZERO)
        {
            // system has no unique solution
            OK = false;
        }

        else
        {
            // start backward substitution
            aVec[aN - 1] = aMatrix[nCols * (n1 - 1) + nCols - 1] / aMatrix[nCols * (n1 - 1) + aN - 1];

            for (int k = 1; k <= NN; k++)
            {
                i            = NN - k + 1;
                const int jj = i + 1;
                const int n2 = nRow[to_size_t(i - 1)];
                float sum    = 0.0f;
                for (int kk = jj; kk <= aN; kk++)
                {
                    sum = sum - aMatrix[nCols * (n2 - 1) + kk - 1] * aVec[kk - 1];
                }
                aVec[i - 1] = (aMatrix[nCols * (n2 - 1) + aN] + sum) / aMatrix[nCols * (n2 - 1) + i - 1];
            }
        }
    }

    return OK;
}
