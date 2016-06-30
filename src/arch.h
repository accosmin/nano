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
#define NANO_UNUSED3(x, y, z) NANO_UNUSED1(x, y); NANO_UNUSED1(z)

// fix "unused variable" warnings (only for release mode)
#ifdef NANO_DEBUG
        #define NANO_UNUSED1_RELEASE(x)
#else
        #define NANO_UNUSED1_RELEASE(x) NANO_UNUSED1(x)
#endif

// string a given variable
#define NANO_STRINGIFY_(x) #x
#define NANO_STRINGIFY(x) NANO_STRINGIFY_(x)

// system information
#if defined(__APPLE__)
        #include <sys/sysctl.h>

        template <typename tinteger>
        tinteger get_sysctl_var(const char* name, const tinteger default_value)
        {
                tinteger value = 0;
                size_t size = sizeof(value);
                return sysctlbyname(name, &value, &size, NULL, 0) ? default_value : value;
        }

        inline auto get_logical_cpus()
        {
                return get_sysctl_var<unsigned int>("hw.logicalcpu", 0);
        }

        inline auto get_physical_cpus()
        {
                return get_sysctl_var<unsigned int>("hw.physicalcpu", 0);
        }

        inline auto get_memsize()
        {
                return get_sysctl_var<unsigned long int>("hw.memsize", 0);
        }

        inline auto get_memsize_gb()
        {
                const unsigned long int giga = 1LL << 30;
                return static_cast<unsigned int>((get_memsize() + giga - 1) / giga);
        }
#endif
