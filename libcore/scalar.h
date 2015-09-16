#pragma once

#include <vector>

namespace ncv
{
        // sizes and indices
        typedef std::size_t                     size_t;
        typedef std::vector<size_t>             sizes_t;
        typedef std::vector<size_t>             indices_t;

        // low-precision scalar
        typedef float                           lscalar_t;
        typedef std::vector<lscalar_t>          lscalars_t;

        // high-precision scalar
        typedef double                          hscalar_t;
        typedef std::vector<hscalar_t>          hscalars_t;

        // default scalar
        typedef hscalar_t                       scalar_t;
        typedef hscalars_t                      scalars_t;
}


