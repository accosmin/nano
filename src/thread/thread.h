#pragma once

#include <thread>
#include <algorithm>

namespace thread
{
        ///
        /// \brief the number of threads available on the system (usually #CPU cores x 2 (HT)).
        ///
        inline unsigned int concurrency()
        {
                return std::max((unsigned int)(1), std::thread::hardware_concurrency());
        }
}
