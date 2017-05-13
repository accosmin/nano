#pragma once

#include "types.h"
#include "ls_step.h"

namespace nano
{
        ///
        /// \brief backtracking line-search,
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition
        ///     see libLBFGS
        ///
        class ls_backtrack_armijo_t
        {
        public:

                ///
                /// \brief compute the current step size
                ///
                ls_step_t operator()(
                        const scalar_t c1, const scalar_t c2,
                        const ls_step_t& step0, const scalar_t t0,
                        const scalar_t decrement = scalar_t(0.5),
                        const scalar_t increment = scalar_t(2.1)) const;
        };

        class ls_backtrack_wolfe_t
        {
        public:

                ///
                /// \brief compute the current step size
                ///
                ls_step_t operator()(
                        const scalar_t c1, const scalar_t c2,
                        const ls_step_t& step0, const scalar_t t0,
                        const scalar_t decrement = scalar_t(0.5),
                        const scalar_t increment = scalar_t(2.1)) const;
        };

        class ls_backtrack_strong_wolfe_t
        {
        public:

                ///
                /// \brief compute the current step size
                ///
                ls_step_t operator()(
                        const scalar_t c1, const scalar_t c2,
                        const ls_step_t& step0, const scalar_t t0,
                        const scalar_t decrement = scalar_t(0.5),
                        const scalar_t increment = scalar_t(2.1)) const;
        };
}

