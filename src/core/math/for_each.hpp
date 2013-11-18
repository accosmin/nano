#ifndef NANOCV_FOREACH_H
#define NANOCV_FOREACH_H

#include <algorithm>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // numerical utility functions for matrices.
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                // transform coefficient-wise a matrix: op(&in)
                template
                <
                        typename tmatrix,
                        typename toperator
                >
                void for_each(tmatrix& in, toperator op)
                {
                        std::for_each(in.data(), in.data() + in.size(), op);
                }
        }
}

#endif // NANOCV_FOREACH_H

