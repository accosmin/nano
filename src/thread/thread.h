#pragma once

#include "arch.h"

namespace thread
{
        ///
        /// \brief the number of threads available on the system (usually #CPU cores x 2 (HT))
        ///
        NANOCV_PUBLIC unsigned int n_threads();

        ///
        /// \brief maximum number of supported threads (by the library)
        ///
        NANOCV_PUBLIC unsigned int max_n_threads();
}
