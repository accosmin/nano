#pragma once

#include <vector>

namespace nano
{
        // sizes and indices
        using size_t = std::size_t;
        using sizes_t = std::vector<size_t>;
        using indices_t = std::vector<size_t>;

        // default scalar
#if defined(NANO_FLOAT_SCALAR)
        using scalar_t = float;
#elif defined(NANO_DOUBLE_SCALAR)
        using scalar_t = double;
#elif defined(NANO_LONG_DOUBLE_SCALAR)
        using scalar_t = long double;
#else
        #error Scalar type not defined! User one of the NANO_[FLOAT|DOUBLE|LONG_DOUBLE]_SCALAR options.
#endif
        using scalars_t = std::vector<scalar_t>;
}


