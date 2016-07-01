#pragma once

#include "types.h"
#include "ls_cubic.h"
#include "ls_bisection.h"
#include "ls_quadratic.h"

namespace nano
{
        ///
        /// \brief interpolation-based line-search (More & Thuente -like?!),
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.60
        ///
        class ls_interpolate_t
        {
        public:

                ///
                /// \brief compute the current step size
                ///
                ls_step_t operator()(
                        const ls_strategy strategy, const scalar_t c1, const scalar_t c2,
                        const ls_step_t& step0, const scalar_t t0) const;

        private:

                ///
                /// \brief zoom-in in the bracketed interval,
                ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.60
                ///
                static ls_step_t zoom(
                        const ls_strategy, const scalar_t c1, const scalar_t c2,
                        const ls_step_t& step0, ls_step_t steplo, ls_step_t stephi);
        };
}

