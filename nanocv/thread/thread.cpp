#include "thread.h"
#include <thread>

namespace ncv
{
        std::size_t n_threads()
        {
                return static_cast<std::size_t>(std::thread::hardware_concurrency());
        }

        std::size_t max_n_threads()
        {
                return n_threads() * 8;
        }
}
