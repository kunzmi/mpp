#include "pixelTypes.h"
#include <ostream>

namespace opp::image
{
std::ostream &operator<<(std::ostream &aOs, const PixelTypeEnum &aPixelType)
{
    switch (aPixelType)
    {
            // double
        case PixelTypeEnum::PTE64fC1:
            aOs << "Pixel64fC1";
            break;
        case PixelTypeEnum::PTE64fC2:
            aOs << "Pixel64fC2";
            break;
        case PixelTypeEnum::PTE64fC3:
            aOs << "Pixel64fC3";
            break;
        case PixelTypeEnum::PTE64fC4:
            aOs << "Pixel64fC4";
            break;
        case PixelTypeEnum::PTE64fC4A:
            aOs << "Pixel64fC4A";
            break;
            // float
        case PixelTypeEnum::PTE32fC1:
            aOs << "Pixel32fC1";
            break;
        case PixelTypeEnum::PTE32fC2:
            aOs << "Pixel32fC2";
            break;
        case PixelTypeEnum::PTE32fC3:
            aOs << "Pixel32fC3";
            break;
        case PixelTypeEnum::PTE32fC4:
            aOs << "Pixel32fC4";
            break;
        case PixelTypeEnum::PTE32fC4A:
            aOs << "Pixel32fC4A";
            break;
            // complex float
        case PixelTypeEnum::PTE32fcC1:
            aOs << "Pixel32fcC1";
            break;
        case PixelTypeEnum::PTE32fcC2:
            aOs << "Pixel32fcC2";
            break;
        case PixelTypeEnum::PTE32fcC3:
            aOs << "Pixel32fcC3";
            break;
        case PixelTypeEnum::PTE32fcC4:
            aOs << "Pixel32fcC4";
            break;
            // int
        case PixelTypeEnum::PTE32sC1:
            aOs << "Pixel32sC1";
            break;
        case PixelTypeEnum::PTE32sC2:
            aOs << "Pixel32sC2";
            break;
        case PixelTypeEnum::PTE32sC3:
            aOs << "Pixel32sC3";
            break;
        case PixelTypeEnum::PTE32sC4:
            aOs << "Pixel32sC4";
            break;
        case PixelTypeEnum::PTE32sC4A:
            aOs << "Pixel32sC4A";
            break;
            // complex int
        case PixelTypeEnum::PTE32scC1:
            aOs << "Pixel32scC1";
            break;
        case PixelTypeEnum::PTE32scC2:
            aOs << "Pixel32scC2";
            break;
        case PixelTypeEnum::PTE32scC3:
            aOs << "Pixel32scC3";
            break;
        case PixelTypeEnum::PTE32scC4:
            aOs << "Pixel32scC4";
            break;
            // unsigned int
        case PixelTypeEnum::PTE32uC1:
            aOs << "Pixel32uC1";
            break;
        case PixelTypeEnum::PTE32uC2:
            aOs << "Pixel32uC2";
            break;
        case PixelTypeEnum::PTE32uC3:
            aOs << "Pixel32uC3";
            break;
        case PixelTypeEnum::PTE32uC4:
            aOs << "Pixel32uC4";
            break;
        case PixelTypeEnum::PTE32uC4A:
            aOs << "Pixel32uC4A";
            break;
            // short
        case PixelTypeEnum::PTE16sC1:
            aOs << "Pixel16sC1";
            break;
        case PixelTypeEnum::PTE16sC2:
            aOs << "Pixel16sC2";
            break;
        case PixelTypeEnum::PTE16sC3:
            aOs << "Pixel16sC3";
            break;
        case PixelTypeEnum::PTE16sC4:
            aOs << "Pixel16sC4";
            break;
        case PixelTypeEnum::PTE16sC4A:
            aOs << "Pixel16sC4A";
            break;
            // complex short
        case PixelTypeEnum::PTE16scC1:
            aOs << "Pixel16scC1";
            break;
        case PixelTypeEnum::PTE16scC2:
            aOs << "Pixel16scC2";
            break;
        case PixelTypeEnum::PTE16scC3:
            aOs << "Pixel16scC3";
            break;
        case PixelTypeEnum::PTE16scC4:
            aOs << "Pixel16scC4";
            break;
            // unsigned short
        case PixelTypeEnum::PTE16uC1:
            aOs << "Pixel16uC1";
            break;
        case PixelTypeEnum::PTE16uC2:
            aOs << "Pixel16uC2";
            break;
        case PixelTypeEnum::PTE16uC3:
            aOs << "Pixel16uC3";
            break;
        case PixelTypeEnum::PTE16uC4:
            aOs << "Pixel16uC4";
            break;
        case PixelTypeEnum::PTE16uC4A:
            aOs << "Pixel16uC4A";
            break;
            // signed byte
        case PixelTypeEnum::PTE8sC1:
            aOs << "Pixel8sC1";
            break;
        case PixelTypeEnum::PTE8sC2:
            aOs << "Pixel8sC2";
            break;
        case PixelTypeEnum::PTE8sC3:
            aOs << "Pixel8sC3";
            break;
        case PixelTypeEnum::PTE8sC4:
            aOs << "Pixel8sC4";
            break;
        case PixelTypeEnum::PTE8sC4A:
            aOs << "Pixel8sC4A";
            break;
            // unsigned byte
        case PixelTypeEnum::PTE8uC1:
            aOs << "Pixel8uC1";
            break;
        case PixelTypeEnum::PTE8uC2:
            aOs << "Pixel8uC2";
            break;
        case PixelTypeEnum::PTE8uC3:
            aOs << "Pixel8uC3";
            break;
        case PixelTypeEnum::PTE8uC4:
            aOs << "Pixel8uC4";
            break;
        case PixelTypeEnum::PTE8uC4A:
            aOs << "Pixel8uC4A";
            break;
            // undefined or unsupported
        default:
            aOs << "Unknown";
            break;
    }
    return aOs;
}
std::wostream &operator<<(std::wostream &aOs, const PixelTypeEnum &aPixelType)
{
    switch (aPixelType)
    {
            // double
        case PixelTypeEnum::PTE64fC1:
            aOs << "Pixel64fC1";
            break;
        case PixelTypeEnum::PTE64fC2:
            aOs << "Pixel64fC2";
            break;
        case PixelTypeEnum::PTE64fC3:
            aOs << "Pixel64fC3";
            break;
        case PixelTypeEnum::PTE64fC4:
            aOs << "Pixel64fC4";
            break;
        case PixelTypeEnum::PTE64fC4A:
            aOs << "Pixel64fC4A";
            break;
            // float
        case PixelTypeEnum::PTE32fC1:
            aOs << "Pixel32fC1";
            break;
        case PixelTypeEnum::PTE32fC2:
            aOs << "Pixel32fC2";
            break;
        case PixelTypeEnum::PTE32fC3:
            aOs << "Pixel32fC3";
            break;
        case PixelTypeEnum::PTE32fC4:
            aOs << "Pixel32fC4";
            break;
        case PixelTypeEnum::PTE32fC4A:
            aOs << "Pixel32fC4A";
            break;
            // complex float
        case PixelTypeEnum::PTE32fcC1:
            aOs << "Pixel32fcC1";
            break;
        case PixelTypeEnum::PTE32fcC2:
            aOs << "Pixel32fcC2";
            break;
        case PixelTypeEnum::PTE32fcC3:
            aOs << "Pixel32fcC3";
            break;
        case PixelTypeEnum::PTE32fcC4:
            aOs << "Pixel32fcC4";
            break;
            // int
        case PixelTypeEnum::PTE32sC1:
            aOs << "Pixel32sC1";
            break;
        case PixelTypeEnum::PTE32sC2:
            aOs << "Pixel32sC2";
            break;
        case PixelTypeEnum::PTE32sC3:
            aOs << "Pixel32sC3";
            break;
        case PixelTypeEnum::PTE32sC4:
            aOs << "Pixel32sC4";
            break;
        case PixelTypeEnum::PTE32sC4A:
            aOs << "Pixel32sC4A";
            break;
            // complex int
        case PixelTypeEnum::PTE32scC1:
            aOs << "Pixel32scC1";
            break;
        case PixelTypeEnum::PTE32scC2:
            aOs << "Pixel32scC2";
            break;
        case PixelTypeEnum::PTE32scC3:
            aOs << "Pixel32scC3";
            break;
        case PixelTypeEnum::PTE32scC4:
            aOs << "Pixel32scC4";
            break;
            // unsigned int
        case PixelTypeEnum::PTE32uC1:
            aOs << "Pixel32uC1";
            break;
        case PixelTypeEnum::PTE32uC2:
            aOs << "Pixel32uC2";
            break;
        case PixelTypeEnum::PTE32uC3:
            aOs << "Pixel32uC3";
            break;
        case PixelTypeEnum::PTE32uC4:
            aOs << "Pixel32uC4";
            break;
        case PixelTypeEnum::PTE32uC4A:
            aOs << "Pixel32uC4A";
            break;
            // short
        case PixelTypeEnum::PTE16sC1:
            aOs << "Pixel16sC1";
            break;
        case PixelTypeEnum::PTE16sC2:
            aOs << "Pixel16sC2";
            break;
        case PixelTypeEnum::PTE16sC3:
            aOs << "Pixel16sC3";
            break;
        case PixelTypeEnum::PTE16sC4:
            aOs << "Pixel16sC4";
            break;
        case PixelTypeEnum::PTE16sC4A:
            aOs << "Pixel16sC4A";
            break;
            // complex short
        case PixelTypeEnum::PTE16scC1:
            aOs << "Pixel16scC1";
            break;
        case PixelTypeEnum::PTE16scC2:
            aOs << "Pixel16scC2";
            break;
        case PixelTypeEnum::PTE16scC3:
            aOs << "Pixel16scC3";
            break;
        case PixelTypeEnum::PTE16scC4:
            aOs << "Pixel16scC4";
            break;
            // unsigned short
        case PixelTypeEnum::PTE16uC1:
            aOs << "Pixel16uC1";
            break;
        case PixelTypeEnum::PTE16uC2:
            aOs << "Pixel16uC2";
            break;
        case PixelTypeEnum::PTE16uC3:
            aOs << "Pixel16uC3";
            break;
        case PixelTypeEnum::PTE16uC4:
            aOs << "Pixel16uC4";
            break;
        case PixelTypeEnum::PTE16uC4A:
            aOs << "Pixel16uC4A";
            break;
            // signed byte
        case PixelTypeEnum::PTE8sC1:
            aOs << "Pixel8sC1";
            break;
        case PixelTypeEnum::PTE8sC2:
            aOs << "Pixel8sC2";
            break;
        case PixelTypeEnum::PTE8sC3:
            aOs << "Pixel8sC3";
            break;
        case PixelTypeEnum::PTE8sC4:
            aOs << "Pixel8sC4";
            break;
        case PixelTypeEnum::PTE8sC4A:
            aOs << "Pixel8sC4A";
            break;
            // unsigned byte
        case PixelTypeEnum::PTE8uC1:
            aOs << "Pixel8uC1";
            break;
        case PixelTypeEnum::PTE8uC2:
            aOs << "Pixel8uC2";
            break;
        case PixelTypeEnum::PTE8uC3:
            aOs << "Pixel8uC3";
            break;
        case PixelTypeEnum::PTE8uC4:
            aOs << "Pixel8uC4";
            break;
        case PixelTypeEnum::PTE8uC4A:
            aOs << "Pixel8uC4A";
            break;
            // undefined or unsupported
        default:
            aOs << "Unknown";
            break;
    }
    return aOs;
}
} // namespace opp::image