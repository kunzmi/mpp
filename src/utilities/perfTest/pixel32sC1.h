#pragma once
#include <common/image/border.h>
#include <ostream>

void runPixel32sC1(size_t aIterations, size_t aRepeats, int aWidth, int aHeight, const mpp::image::Border &aBorder,
                   std::ofstream &aCsv);