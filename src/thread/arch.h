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
