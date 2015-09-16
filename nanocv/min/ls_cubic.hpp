#pragma once

#include "cubic.hpp"
#include <utility>

namespace ncv
{
        namespace min
        {
                ///
                /// \brief cubic interpolation in the [step0, step1] line-search interval
                ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
                ///
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar
                >
                std::pair<tscalar, tscalar> ls_cubic(const tstep& step0, const tstep& step1)
                {
                        const tscalar x0 = step0.alpha(), f0 = step0.phi(), g0 = step0.gphi();
                        const tscalar x1 = step1.alpha(), f1 = step1.phi(), g1 = step1.gphi();

                        const cubic_t<tscalar> c(x0, f0, g0, x1, f1, g1);

                        tscalar min1, min2;
                        c.extremum(min1, min2);

                        return std::make_pair(min1, min2);
                }
        }
}

