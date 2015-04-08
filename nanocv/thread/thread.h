#pragma once

#include <thread>
#include <utility>

namespace ncv
{
        ///
        /// \brief the number of threads available on the system
        ///
        inline std::size_t n_threads()
        {
                return static_cast<std::size_t>(std::thread::hardware_concurrency());
        }

        ///
        /// \brief maximum number of supported threads
        ///
        inline std::size_t max_n_threads()
        {
                return n_threads() * 8;
        }
}
