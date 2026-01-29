#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wuseless-cast"

#endif // defined(__GNUC__)

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"

#endif // defined(__clang__)

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4068) // warning unknown pragma
#endif
