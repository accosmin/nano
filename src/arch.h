#pragma once

// export symbols in shared libraries
#if defined _WIN32 || defined __CYGWIN__
        #ifdef __GNUC__
                #define NANO_PUBLIC __attribute__ ((dllexport))
        #else
                #define NANO_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
        #endif
        #define NANO_PRIVATE
#else
        #if __GNUC__ >= 4
                #define NANO_PUBLIC __attribute__ ((visibility ("default")))
                #define NANO_PRIVATE  __attribute__ ((visibility ("hidden")))
        #else
                #define NANO_PUBLIC
                #define NANO_PRIVATE
        #endif
#endif

// fix "unused variable" warnings
#define NANO_UNUSED1(x) (void)(x)
#define NANO_UNUSED2(x, y) NANO_UNUSED1(x); NANO_UNUSED1(y)
#define NANO_UNUSED3(x, y, z) NANO_UNUSED2(x, y); NANO_UNUSED1(z)

// fix "unused variable" warnings (only for release mode)
#ifdef NANO_DEBUG
        #define NANO_UNUSED1_RELEASE(x)
        #define NANO_UNUSED2_RELEASE(x, y)
        #define NANO_UNUSED3_RELEASE(x, y, z)
#else
        #define NANO_UNUSED1_RELEASE(x) NANO_UNUSED1(x)
        #define NANO_UNUSED2_RELEASE(x, y) NANO_UNUSED2(x, y)
        #define NANO_UNUSED3_RELEASE(x, y, z) NANO_UNUSED3(x, y, z)
#endif

// string a given variable
#define NANO_STRINGIFY_(x) #x
#define NANO_STRINGIFY(x) NANO_STRINGIFY_(x)

namespace nano
{
        // system information
        NANO_PUBLIC unsigned int logical_cpus();
        NANO_PUBLIC unsigned int physical_cpus();
        NANO_PUBLIC unsigned long long int memsize();

        inline unsigned int memsize_gb()
        {
                const unsigned long long int giga = 1LL << 30;
                return static_cast<unsigned int>((memsize() + giga - 1) / giga);
        }
}
