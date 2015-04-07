#pragma once

// aliasing (may speed-up some array-based operations)
#if defined(__GNUC__) && ((__GNUC__ > 3) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
        #define NANOCV_RESTRICT __restrict
#elif defined(_MSC_VER) && _MSC_VER >= 1400
        #define NANOCV_RESTRICT __restrict
#else
        #define NANOCV_RESTRICT
#endif

// export symbols in shared libraries
#if defined _WIN32 || defined __CYGWIN__
        #ifdef __GNUC__
                #define NANOCV_PUBLIC __attribute__ ((dllexport))
        #else
                #define NANOCV_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
        #endif
        #define NANOCV_PRIVATE
#else
        #if __GNUC__ >= 4
                #define NANOCV_PUBLIC __attribute__ ((visibility ("default")))
                #define NANOCV_PRIVATE  __attribute__ ((visibility ("hidden")))
        #else
                #define NANOCV_PUBLIC
                #define NANOCV_PRIVATE
        #endif
#endif
