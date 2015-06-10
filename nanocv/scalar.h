#pragma once

#include <vector>

namespace ncv
{
        // numerical types
        typedef std::size_t                                     size_t;        
        typedef std::vector<size_t>                             sizes_t;
        typedef std::vector<size_t>                             indices_t;

#ifdef NANOCV_WITH_FLOAT
        typedef float                                           scalar_t;
#elseif MANOCV_WITH_DOUBLE
        typedef double                                          scalar_t;
#elseif NANOCV_WIDTH_LONG_DOUBLE
        typedef long double                                     scalar_t;
#else
        typedef double                                          scalar_t;
#endif
        typedef std::vector<scalar_t>                           scalars_t;
}


