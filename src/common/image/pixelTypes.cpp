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
            aOs << pixel_type_name<Pixel64fC1>::value;
            break;
        case PixelTypeEnum::PTE64fC2:
            aOs << pixel_type_name<Pixel64fC2>::value;
            break;
        case PixelTypeEnum::PTE64fC3:
            aOs << pixel_type_name<Pixel64fC3>::value;
            break;
        case PixelTypeEnum::PTE64fC4:
            aOs << pixel_type_name<Pixel64fC4>::value;
            break;
        case PixelTypeEnum::PTE64fC4A:
            aOs << pixel_type_name<Pixel64fC4A>::value;
            break;
            // double complex
        case PixelTypeEnum::PTE64fcC1:
            aOs << pixel_type_name<Pixel64fcC1>::value;
            break;
        case PixelTypeEnum::PTE64fcC2:
            aOs << pixel_type_name<Pixel64fcC2>::value;
            break;
        case PixelTypeEnum::PTE64fcC3:
            aOs << pixel_type_name<Pixel64fcC3>::value;
            break;
        case PixelTypeEnum::PTE64fcC4:
            aOs << pixel_type_name<Pixel64fcC4>::value;
            break;
            // float
        case PixelTypeEnum::PTE32fC1:
            aOs << pixel_type_name<Pixel32fC1>::value;
            break;
        case PixelTypeEnum::PTE32fC2:
            aOs << pixel_type_name<Pixel32fC2>::value;
            break;
        case PixelTypeEnum::PTE32fC3:
            aOs << pixel_type_name<Pixel32fC3>::value;
            break;
        case PixelTypeEnum::PTE32fC4:
            aOs << pixel_type_name<Pixel32fC4>::value;
            break;
        case PixelTypeEnum::PTE32fC4A:
            aOs << pixel_type_name<Pixel32fC4A>::value;
            break;
            // complex float
        case PixelTypeEnum::PTE32fcC1:
            aOs << pixel_type_name<Pixel32fcC1>::value;
            break;
        case PixelTypeEnum::PTE32fcC2:
            aOs << pixel_type_name<Pixel32fcC2>::value;
            break;
        case PixelTypeEnum::PTE32fcC3:
            aOs << pixel_type_name<Pixel32fcC3>::value;
            break;
        case PixelTypeEnum::PTE32fcC4:
            aOs << pixel_type_name<Pixel32fcC4>::value;
            break;
            // float16
        case PixelTypeEnum::PTE16fC1:
            aOs << pixel_type_name<Pixel16fC1>::value;
            break;
        case PixelTypeEnum::PTE16fC2:
            aOs << pixel_type_name<Pixel16fC2>::value;
            break;
        case PixelTypeEnum::PTE16fC3:
            aOs << pixel_type_name<Pixel16fC3>::value;
            break;
        case PixelTypeEnum::PTE16fC4:
            aOs << pixel_type_name<Pixel16fC4>::value;
            break;
        case PixelTypeEnum::PTE16fC4A:
            aOs << pixel_type_name<Pixel16fC4A>::value;
            break;
            // float16 complex
        case PixelTypeEnum::PTE16fcC1:
            aOs << pixel_type_name<Pixel16fcC1>::value;
            break;
        case PixelTypeEnum::PTE16fcC2:
            aOs << pixel_type_name<Pixel16fcC2>::value;
            break;
        case PixelTypeEnum::PTE16fcC3:
            aOs << pixel_type_name<Pixel16fcC3>::value;
            break;
        case PixelTypeEnum::PTE16fcC4:
            aOs << pixel_type_name<Pixel16fcC4>::value;
            break;
            // bfloat16
        case PixelTypeEnum::PTE16bfC1:
            aOs << pixel_type_name<Pixel16bfC1>::value;
            break;
        case PixelTypeEnum::PTE16bfC2:
            aOs << pixel_type_name<Pixel16bfC2>::value;
            break;
        case PixelTypeEnum::PTE16bfC3:
            aOs << pixel_type_name<Pixel16bfC3>::value;
            break;
        case PixelTypeEnum::PTE16bfC4:
            aOs << pixel_type_name<Pixel16bfC4>::value;
            break;
        case PixelTypeEnum::PTE16bfC4A:
            aOs << pixel_type_name<Pixel16bfC4A>::value;
            break;
            // bfloat16 complex
        case PixelTypeEnum::PTE16bfcC1:
            aOs << pixel_type_name<Pixel16bfcC1>::value;
            break;
        case PixelTypeEnum::PTE16bfcC2:
            aOs << pixel_type_name<Pixel16bfcC2>::value;
            break;
        case PixelTypeEnum::PTE16bfcC3:
            aOs << pixel_type_name<Pixel16bfcC3>::value;
            break;
        case PixelTypeEnum::PTE16bfcC4:
            aOs << pixel_type_name<Pixel16bfcC4>::value;
            break;
            // int
        case PixelTypeEnum::PTE32sC1:
            aOs << pixel_type_name<Pixel32sC1>::value;
            break;
        case PixelTypeEnum::PTE32sC2:
            aOs << pixel_type_name<Pixel32sC2>::value;
            break;
        case PixelTypeEnum::PTE32sC3:
            aOs << pixel_type_name<Pixel32sC3>::value;
            break;
        case PixelTypeEnum::PTE32sC4:
            aOs << pixel_type_name<Pixel32sC4>::value;
            break;
        case PixelTypeEnum::PTE32sC4A:
            aOs << pixel_type_name<Pixel32sC4A>::value;
            break;
            // complex int
        case PixelTypeEnum::PTE32scC1:
            aOs << pixel_type_name<Pixel32scC1>::value;
            break;
        case PixelTypeEnum::PTE32scC2:
            aOs << pixel_type_name<Pixel32scC2>::value;
            break;
        case PixelTypeEnum::PTE32scC3:
            aOs << pixel_type_name<Pixel32scC3>::value;
            break;
        case PixelTypeEnum::PTE32scC4:
            aOs << pixel_type_name<Pixel32scC4>::value;
            break;
            // unsigned int
        case PixelTypeEnum::PTE32uC1:
            aOs << pixel_type_name<Pixel32uC1>::value;
            break;
        case PixelTypeEnum::PTE32uC2:
            aOs << pixel_type_name<Pixel32uC2>::value;
            break;
        case PixelTypeEnum::PTE32uC3:
            aOs << pixel_type_name<Pixel32uC3>::value;
            break;
        case PixelTypeEnum::PTE32uC4:
            aOs << pixel_type_name<Pixel32uC4>::value;
            break;
        case PixelTypeEnum::PTE32uC4A:
            aOs << pixel_type_name<Pixel32uC4A>::value;
            break;
            // short
        case PixelTypeEnum::PTE16sC1:
            aOs << pixel_type_name<Pixel16sC1>::value;
            break;
        case PixelTypeEnum::PTE16sC2:
            aOs << pixel_type_name<Pixel16sC2>::value;
            break;
        case PixelTypeEnum::PTE16sC3:
            aOs << pixel_type_name<Pixel16sC3>::value;
            break;
        case PixelTypeEnum::PTE16sC4:
            aOs << pixel_type_name<Pixel16sC4>::value;
            break;
        case PixelTypeEnum::PTE16sC4A:
            aOs << pixel_type_name<Pixel16sC4A>::value;
            break;
            // complex short
        case PixelTypeEnum::PTE16scC1:
            aOs << pixel_type_name<Pixel16scC1>::value;
            break;
        case PixelTypeEnum::PTE16scC2:
            aOs << pixel_type_name<Pixel16scC2>::value;
            break;
        case PixelTypeEnum::PTE16scC3:
            aOs << pixel_type_name<Pixel16scC3>::value;
            break;
        case PixelTypeEnum::PTE16scC4:
            aOs << pixel_type_name<Pixel16scC4>::value;
            break;
            // unsigned short
        case PixelTypeEnum::PTE16uC1:
            aOs << pixel_type_name<Pixel16uC1>::value;
            break;
        case PixelTypeEnum::PTE16uC2:
            aOs << pixel_type_name<Pixel16uC2>::value;
            break;
        case PixelTypeEnum::PTE16uC3:
            aOs << pixel_type_name<Pixel16uC3>::value;
            break;
        case PixelTypeEnum::PTE16uC4:
            aOs << pixel_type_name<Pixel16uC4>::value;
            break;
        case PixelTypeEnum::PTE16uC4A:
            aOs << pixel_type_name<Pixel16uC4A>::value;
            break;
            // signed byte
        case PixelTypeEnum::PTE8sC1:
            aOs << pixel_type_name<Pixel8sC1>::value;
            break;
        case PixelTypeEnum::PTE8sC2:
            aOs << pixel_type_name<Pixel8sC2>::value;
            break;
        case PixelTypeEnum::PTE8sC3:
            aOs << pixel_type_name<Pixel8sC3>::value;
            break;
        case PixelTypeEnum::PTE8sC4:
            aOs << pixel_type_name<Pixel8sC4>::value;
            break;
        case PixelTypeEnum::PTE8sC4A:
            aOs << pixel_type_name<Pixel8sC4A>::value;
            break;
            // unsigned byte
        case PixelTypeEnum::PTE8uC1:
            aOs << pixel_type_name<Pixel8uC1>::value;
            break;
        case PixelTypeEnum::PTE8uC2:
            aOs << pixel_type_name<Pixel8uC2>::value;
            break;
        case PixelTypeEnum::PTE8uC3:
            aOs << pixel_type_name<Pixel8uC3>::value;
            break;
        case PixelTypeEnum::PTE8uC4:
            aOs << pixel_type_name<Pixel8uC4>::value;
            break;
        case PixelTypeEnum::PTE8uC4A:
            aOs << pixel_type_name<Pixel8uC4A>::value;
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
            aOs << pixel_type_name<Pixel64fC1>::value;
            break;
        case PixelTypeEnum::PTE64fC2:
            aOs << pixel_type_name<Pixel64fC2>::value;
            break;
        case PixelTypeEnum::PTE64fC3:
            aOs << pixel_type_name<Pixel64fC3>::value;
            break;
        case PixelTypeEnum::PTE64fC4:
            aOs << pixel_type_name<Pixel64fC4>::value;
            break;
        case PixelTypeEnum::PTE64fC4A:
            aOs << pixel_type_name<Pixel64fC4A>::value;
            break;
            // double complex
        case PixelTypeEnum::PTE64fcC1:
            aOs << pixel_type_name<Pixel64fcC1>::value;
            break;
        case PixelTypeEnum::PTE64fcC2:
            aOs << pixel_type_name<Pixel64fcC2>::value;
            break;
        case PixelTypeEnum::PTE64fcC3:
            aOs << pixel_type_name<Pixel64fcC3>::value;
            break;
        case PixelTypeEnum::PTE64fcC4:
            aOs << pixel_type_name<Pixel64fcC4>::value;
            break;
            // float
        case PixelTypeEnum::PTE32fC1:
            aOs << pixel_type_name<Pixel32fC1>::value;
            break;
        case PixelTypeEnum::PTE32fC2:
            aOs << pixel_type_name<Pixel32fC2>::value;
            break;
        case PixelTypeEnum::PTE32fC3:
            aOs << pixel_type_name<Pixel32fC3>::value;
            break;
        case PixelTypeEnum::PTE32fC4:
            aOs << pixel_type_name<Pixel32fC4>::value;
            break;
        case PixelTypeEnum::PTE32fC4A:
            aOs << pixel_type_name<Pixel32fC4A>::value;
            break;
            // complex float
        case PixelTypeEnum::PTE32fcC1:
            aOs << pixel_type_name<Pixel32fcC1>::value;
            break;
        case PixelTypeEnum::PTE32fcC2:
            aOs << pixel_type_name<Pixel32fcC2>::value;
            break;
        case PixelTypeEnum::PTE32fcC3:
            aOs << pixel_type_name<Pixel32fcC3>::value;
            break;
        case PixelTypeEnum::PTE32fcC4:
            aOs << pixel_type_name<Pixel32fcC4>::value;
            break;
            // float16
        case PixelTypeEnum::PTE16fC1:
            aOs << pixel_type_name<Pixel16fC1>::value;
            break;
        case PixelTypeEnum::PTE16fC2:
            aOs << pixel_type_name<Pixel16fC2>::value;
            break;
        case PixelTypeEnum::PTE16fC3:
            aOs << pixel_type_name<Pixel16fC3>::value;
            break;
        case PixelTypeEnum::PTE16fC4:
            aOs << pixel_type_name<Pixel16fC4>::value;
            break;
        case PixelTypeEnum::PTE16fC4A:
            aOs << pixel_type_name<Pixel16fC4A>::value;
            break;
            // float16 complex
        case PixelTypeEnum::PTE16fcC1:
            aOs << pixel_type_name<Pixel16fcC1>::value;
            break;
        case PixelTypeEnum::PTE16fcC2:
            aOs << pixel_type_name<Pixel16fcC2>::value;
            break;
        case PixelTypeEnum::PTE16fcC3:
            aOs << pixel_type_name<Pixel16fcC3>::value;
            break;
        case PixelTypeEnum::PTE16fcC4:
            aOs << pixel_type_name<Pixel16fcC4>::value;
            break;
            // bfloat16
        case PixelTypeEnum::PTE16bfC1:
            aOs << pixel_type_name<Pixel16bfC1>::value;
            break;
        case PixelTypeEnum::PTE16bfC2:
            aOs << pixel_type_name<Pixel16bfC2>::value;
            break;
        case PixelTypeEnum::PTE16bfC3:
            aOs << pixel_type_name<Pixel16bfC3>::value;
            break;
        case PixelTypeEnum::PTE16bfC4:
            aOs << pixel_type_name<Pixel16bfC4>::value;
            break;
        case PixelTypeEnum::PTE16bfC4A:
            aOs << pixel_type_name<Pixel16bfC4A>::value;
            break;
            // bfloat16 complex
        case PixelTypeEnum::PTE16bfcC1:
            aOs << pixel_type_name<Pixel16bfcC1>::value;
            break;
        case PixelTypeEnum::PTE16bfcC2:
            aOs << pixel_type_name<Pixel16bfcC2>::value;
            break;
        case PixelTypeEnum::PTE16bfcC3:
            aOs << pixel_type_name<Pixel16bfcC3>::value;
            break;
        case PixelTypeEnum::PTE16bfcC4:
            aOs << pixel_type_name<Pixel16bfcC4>::value;
            break;
            // int
        case PixelTypeEnum::PTE32sC1:
            aOs << pixel_type_name<Pixel32sC1>::value;
            break;
        case PixelTypeEnum::PTE32sC2:
            aOs << pixel_type_name<Pixel32sC2>::value;
            break;
        case PixelTypeEnum::PTE32sC3:
            aOs << pixel_type_name<Pixel32sC3>::value;
            break;
        case PixelTypeEnum::PTE32sC4:
            aOs << pixel_type_name<Pixel32sC4>::value;
            break;
        case PixelTypeEnum::PTE32sC4A:
            aOs << pixel_type_name<Pixel32sC4A>::value;
            break;
            // complex int
        case PixelTypeEnum::PTE32scC1:
            aOs << pixel_type_name<Pixel32scC1>::value;
            break;
        case PixelTypeEnum::PTE32scC2:
            aOs << pixel_type_name<Pixel32scC2>::value;
            break;
        case PixelTypeEnum::PTE32scC3:
            aOs << pixel_type_name<Pixel32scC3>::value;
            break;
        case PixelTypeEnum::PTE32scC4:
            aOs << pixel_type_name<Pixel32scC4>::value;
            break;
            // unsigned int
        case PixelTypeEnum::PTE32uC1:
            aOs << pixel_type_name<Pixel32uC1>::value;
            break;
        case PixelTypeEnum::PTE32uC2:
            aOs << pixel_type_name<Pixel32uC2>::value;
            break;
        case PixelTypeEnum::PTE32uC3:
            aOs << pixel_type_name<Pixel32uC3>::value;
            break;
        case PixelTypeEnum::PTE32uC4:
            aOs << pixel_type_name<Pixel32uC4>::value;
            break;
        case PixelTypeEnum::PTE32uC4A:
            aOs << pixel_type_name<Pixel32uC4A>::value;
            break;
            // short
        case PixelTypeEnum::PTE16sC1:
            aOs << pixel_type_name<Pixel16sC1>::value;
            break;
        case PixelTypeEnum::PTE16sC2:
            aOs << pixel_type_name<Pixel16sC2>::value;
            break;
        case PixelTypeEnum::PTE16sC3:
            aOs << pixel_type_name<Pixel16sC3>::value;
            break;
        case PixelTypeEnum::PTE16sC4:
            aOs << pixel_type_name<Pixel16sC4>::value;
            break;
        case PixelTypeEnum::PTE16sC4A:
            aOs << pixel_type_name<Pixel16sC4A>::value;
            break;
            // complex short
        case PixelTypeEnum::PTE16scC1:
            aOs << pixel_type_name<Pixel16scC1>::value;
            break;
        case PixelTypeEnum::PTE16scC2:
            aOs << pixel_type_name<Pixel16scC2>::value;
            break;
        case PixelTypeEnum::PTE16scC3:
            aOs << pixel_type_name<Pixel16scC3>::value;
            break;
        case PixelTypeEnum::PTE16scC4:
            aOs << pixel_type_name<Pixel16scC4>::value;
            break;
            // unsigned short
        case PixelTypeEnum::PTE16uC1:
            aOs << pixel_type_name<Pixel16uC1>::value;
            break;
        case PixelTypeEnum::PTE16uC2:
            aOs << pixel_type_name<Pixel16uC2>::value;
            break;
        case PixelTypeEnum::PTE16uC3:
            aOs << pixel_type_name<Pixel16uC3>::value;
            break;
        case PixelTypeEnum::PTE16uC4:
            aOs << pixel_type_name<Pixel16uC4>::value;
            break;
        case PixelTypeEnum::PTE16uC4A:
            aOs << pixel_type_name<Pixel16uC4A>::value;
            break;
            // signed byte
        case PixelTypeEnum::PTE8sC1:
            aOs << pixel_type_name<Pixel8sC1>::value;
            break;
        case PixelTypeEnum::PTE8sC2:
            aOs << pixel_type_name<Pixel8sC2>::value;
            break;
        case PixelTypeEnum::PTE8sC3:
            aOs << pixel_type_name<Pixel8sC3>::value;
            break;
        case PixelTypeEnum::PTE8sC4:
            aOs << pixel_type_name<Pixel8sC4>::value;
            break;
        case PixelTypeEnum::PTE8sC4A:
            aOs << pixel_type_name<Pixel8sC4A>::value;
            break;
            // unsigned byte
        case PixelTypeEnum::PTE8uC1:
            aOs << pixel_type_name<Pixel8uC1>::value;
            break;
        case PixelTypeEnum::PTE8uC2:
            aOs << pixel_type_name<Pixel8uC2>::value;
            break;
        case PixelTypeEnum::PTE8uC3:
            aOs << pixel_type_name<Pixel8uC3>::value;
            break;
        case PixelTypeEnum::PTE8uC4:
            aOs << pixel_type_name<Pixel8uC4>::value;
            break;
        case PixelTypeEnum::PTE8uC4A:
            aOs << pixel_type_name<Pixel8uC4A>::value;
            break;
            // undefined or unsupported
        default:
            aOs << "Unknown";
            break;
    }
    return aOs;
}
} // namespace opp::image