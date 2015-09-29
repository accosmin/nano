#pragma once

namespace min
{
        ///
        /// \brief bisection interpolation in the [step0, step1] line-search interval
        ///
        template
        <
                typename tstep,
                typename tscalar = typename tstep::tscalar
        >
        tscalar ls_bisection(const tstep& step0, const tstep& step1)
        {
                return (step0.alpha() + step1.alpha()) / 2;
        }
}

