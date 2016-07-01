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
unsigned int get_logical_cpus();
unsigned int get_physical_cpus();
unsigned long long int get_memsize();

inline unsigned int get_memsize_gb()
{
        const unsigned long long int giga = 1LL << 30;
        return static_cast<unsigned int>((get_memsize() + giga - 1) / giga);
}

#if defined(__APPLE__)
        #include <sys/sysctl.h>

        template <typename tinteger>
        tinteger get_sysctl_var(const char* name, const tinteger default_value)
        {
                tinteger value = 0;
                size_t size = sizeof(value);
                return sysctlbyname(name, &value, &size, nullptr, 0) ? default_value : value;
        }

        inline unsigned int get_logical_cpus()
        {
                return get_sysctl_var<unsigned int>("hw.logicalcpu", 0);
        }

        inline unsigned int get_physical_cpus()
        {
                return get_sysctl_var<unsigned int>("hw.physicalcpu", 0);
        }

        inline unsigned long long int get_memsize()
        {
                return get_sysctl_var<unsigned long long int>("hw.memsize", 0);
        }

#elif defined(__linux__)
        #include <cstdlib>
        #include <string>
        #include <fstream>

        template <typename tinteger>
        tinteger get_command_value(const char* command, const tinteger default_value)
        {
                static const char* filename = "/tmp/cmd.txt";

                const auto ret = std::system((std::string(command) + " > " + filename).c_str());
                NANO_UNUSED1(ret);

                tinteger value = default_value;
                std::ifstream is(filename);
                is >> value;
                return value;
        }

        inline unsigned int get_logical_cpus()
        {
                return get_command_value<unsigned int>("grep processor /proc/cpuinfo | wc -l", 0);
        }

        inline unsigned int get_physical_cpus()
        {
                return get_command_value<unsigned int>("grep cores /proc/cpuinfo | cut -d ':' -f 2 | sort -u", 0);
        }

        inline unsigned long long int get_memsize()
        {
                return get_command_value<unsigned long long int>("grep MemTotal /proc/meminfo | tr -s ' ' | cut -d ' ' -f 2", 0) * 1024;
        }
#endif
