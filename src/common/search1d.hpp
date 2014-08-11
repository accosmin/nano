#ifndef NANOCV_SEARCH1D_H
#define NANOCV_SEARCH1D_H

#include <cmath>
#include <limits>

namespace ncv
{
        ///
        /// \brief search for a 1D parameter that minimizes a given operator, 
        ///     using a greedy approach on the logarithmic scale.
        ///
        template
        <
                typename toperator,     ///< toperator(tscalar param) returns a tscalar score
                typename tscalar
        >
        tscalar min_search1d(const toperator& op, tscalar minlog, tscalar maxlog, tscalar epslog)
        {
                tscalar bestlog = (maxlog + minlog) / 2;
                
                tscalar bestret = std::numeric_limits<tscalar>::max();
                
                tscalar dist = maxlog - minlog;
                while (dist > epslog && epslog > tscalar(0))
                {
                        
                }
                
                return bestlog;
        }
}

#endif // NANOCV_SEARCH1D_H

