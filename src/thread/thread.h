#pragma once

#include <thread>
#include <algorithm>

namespace nano
{
        ///
        /// \brief the number of threads available on the system (usually #CPU cores x 2 (HT))
        ///
        inline unsigned int n_threads()
        {
                return std::max((unsigned int)(1), std::thread::hardware_concurrency());
        }
}
