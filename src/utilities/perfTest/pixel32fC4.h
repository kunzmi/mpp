#pragma once
#include <common/image/border.h>
#include <ostream>

void runPixel32fC4(size_t aIterations, size_t aRepeats, int aWidth, int aHeight, const opp::image::Border &aBorder,
                   std::ofstream &aCsv);