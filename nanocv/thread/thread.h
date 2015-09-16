#pragma once

#include <utility>
#include "core/arch.h"

namespace ncv
{
        ///
        /// \brief the number of threads available on the system
        ///
        NANOCV_PUBLIC std::size_t n_threads();

        ///
        /// \brief maximum number of supported threads
        ///
        NANOCV_PUBLIC std::size_t max_n_threads();
}
