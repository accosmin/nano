#pragma once

// export symbols in shared libraries
#if defined _WIN32 || defined __CYGWIN__
        #ifdef __GNUC__
                #define THREAD_PUBLIC __attribute__ ((dllexport))
        #else
                #define THREAD_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
        #endif
        #define THREAD_PRIVATE
#else
        #if __GNUC__ >= 4
                #define THREAD_PUBLIC __attribute__ ((visibility ("default")))
                #define THREAD_PRIVATE  __attribute__ ((visibility ("hidden")))
        #else
                #define THREAD_PUBLIC
                #define THREAD_PRIVATE
        #endif
#endif

// fix "unused variable" warnings
#define THREAD_UNUSED1(x) (void)(x)
#define THREAD_UNUSED2(x, y) THREAD_UNUSED1(x); THREAD_UNUSED1(y)
#define THREAD_UNUSED3(x, y, z) THREAD_UNUSED1(x); THREAD_UNUSED1(y); THREAD_UNUSED1(z)

// fix "unused variable" warnings (only for release mode)
#ifdef THREAD_DEBUG
        #define THREAD_UNUSED1_RELEASE(x)
#else
        #define THREAD_UNUSED1_RELEASE(x) THREAD_UNUSED1(x)
#endif
