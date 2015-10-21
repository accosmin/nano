#include "thread.h"
#include <thread>

namespace thread
{
        unsigned int n_threads()
        {
                return std::thread::hardware_concurrency();
        }

        unsigned int max_n_threads()
        {
                return n_threads() * 8;
        }
}
