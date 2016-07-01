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

#if defined(__APPLE__)
        #include <sys/sysctl.h>
#elif defined(__linux__)
        #include <unistd.h>
        #include <sys/sysinfo.h>
#else
        #error Unsupported platform
#endif

// system information
namespace nano
{
        unsigned int logical_cpus();
        unsigned int physical_cpus();
        unsigned long long int memsize();

        inline unsigned int memsize_gb()
        {
                const unsigned long long int giga = 1LL << 30;
                return static_cast<unsigned int>((memsize() + giga - 1) / giga);
        }

#if defined(__APPLE__)
        template <typename tinteger>
        tinteger sysctl_var(const char* name, const tinteger default_value)
        {
                tinteger value = 0;
                size_t size = sizeof(value);
                return sysctlbyname(name, &value, &size, nullptr, 0) ? default_value : value;
        }

        inline unsigned int logical_cpus()
        {
                return sysctl_var<unsigned int>("hw.logicalcpu", 0);
        }

        inline unsigned int physical_cpus()
        {
                return sysctl_var<unsigned int>("hw.physicalcpu", 0);
        }

        inline unsigned long long int memsize()
        {
                return sysctl_var<unsigned long long int>("hw.memsize", 0);
        }

#elif defined(__linux__)
        inline unsigned int logical_cpus()
        {
                return (unsigned int)sysconf(_SC_NPROCESSORS_ONLN);
        }

        inline unsigned int physical_cpus()
        {
                unsigned int registers[4];
                __asm__ __volatile__ ("cpuid " :
                      "=a" (registers[0]),
                      "=b" (registers[1]),
                      "=c" (registers[2]),
                      "=d" (registers[3])
                      : "a" (1), "c" (0));
                const unsigned CPUFeatureSet = registers[3];
                const bool hyperthreading = CPUFeatureSet & (1 << 28);
                return hyperthreading ? (logical_cpus() / 2) : logical_cpus();
        }

        inline unsigned long long int memsize()
        {
                struct sysinfo info;
                sysinfo(&info);
                return (unsigned long long int)info.totalram * (unsigned long long int)info.mem_unit;
        }
#endif
}
