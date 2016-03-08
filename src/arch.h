#pragma once

// export symbols in shared libraries
#if defined _WIN32 || defined __CYGWIN__
        #ifdef __GNUC__
                #define ZOB_PUBLIC __attribute__ ((dllexport))
        #else
                #define ZOB_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
        #endif
        #define ZOB_PRIVATE
#else
        #if __GNUC__ >= 4
                #define ZOB_PUBLIC __attribute__ ((visibility ("default")))
                #define ZOB_PRIVATE  __attribute__ ((visibility ("hidden")))
        #else
                #define ZOB_PUBLIC
                #define ZOB_PRIVATE
        #endif
#endif

// fix "unused variable" warnings
#define ZOB_UNUSED1(x) (void)(x)
#define ZOB_UNUSED2(x, y) ZOB_UNUSED1(x); ZOB_UNUSED1(y)
#define ZOB_UNUSED3(x, y, z) ZOB_UNUSED1(x); ZOB_UNUSED1(y); ZOB_UNUSED1(z)

// fix "unused variable" warnings (only for release mode)
#ifdef ZOB_DEBUG
        #define ZOB_UNUSED1_RELEASE(x)
#else
        #define ZOB_UNUSED1_RELEASE(x) ZOB_UNUSED1(x)
#endif

// string a given variable
#define ZOB_STRINGIFY_(x) #x
#define ZOB_STRINGIFY(x) ZOB_STRINGIFY_(x)
