#pragma once

#if defined(__GNUC__) && ((__GNUC__ > 3) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
        #define NANOCV_RESTRICT __restrict
#elif defined(_MSC_VER) && _MSC_VER >= 1400
        #define NANOCV_RESTRICT __restrict
#else
        #define NANOCV_RESTRICT
#endif
