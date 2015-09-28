#pragma once

#include "linesearch.h"
#include "ls_step.hpp"

#include <iomanip>
#include <iostream>

namespace ncv
{
        namespace min
        {
                ///
                /// \brief backtracking line-search,
                ///     see "Numerical optimization", Nocedal & Wright, 2nd edition
                ///     see libLBFGS
                ///
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar,
                        typename tsize = typename tstep::tsize
                >
                class linesearch_backtracking_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        linesearch_backtracking_t()
                        {
                        }

                        ///
                        /// \brief compute the current step size
                        ///
                        tstep operator()(
                                const ls_strategy strategy, const tscalar c1, const tscalar c2,
                                const tstep& step0, const tscalar t0,
                                const tscalar decrement = tscalar(0.5),
                                const tscalar increment = tscalar(2.1),
                                const tsize max_iters = 64) const
                        {
                                tstep step(step0);
                                tscalar t = t0;

                                for (tsize i = 0; i < max_iters; i ++)
                                {
                                        std::cout << std::setprecision(12)
                                                  << "ls_backtracking [" << i << "/" << max_iters
                                                  << "]: t = " << t << "/" << t0 << "\n";

                                        if (!step.reset(t))
                                        {
                                                std::cout << "ls_backtracking: cannot reset step!\n";

                                                return step0;
                                        }

                                        // check Armijo condition
                                        if (!step.has_armijo(c1))
                                        {
                                                std::cout << std::setprecision(12)
                                                          << "t = " << t
                                                          << ", phi = " << step.phi()
                                                          << " < " << (step.phi0() + step.alpha() * c1 * step.gphi0())
                                                          << ", phi0 = " << step.phi0()
                                                          << ", gphi0 = " << step.gphi0()
                                                          << "\n";

                                                t *= decrement;
                                        }
                                        else
                                        {
                                                if (strategy == ls_strategy::backtrack_armijo)
                                                {
                                                        std::cout << "ls_backtracking: exit armijo\n";

                                                        return step;
                                                }

                                                // check Wolfe condition
                                                if (!step.has_wolfe(c2))
                                                {
                                                        t *= increment;
                                                }
                                                else
                                                {
                                                        if (strategy == ls_strategy::backtrack_wolfe)
                                                        {
                                                                std::cout << "ls_backtracking: exit wolfe\n";

                                                                return step;
                                                        }

                                                        // check strong Wolfe condition
                                                        if (!step.has_strong_wolfe(c2))
                                                        {
                                                                t *= decrement;
                                                        }
                                                        else
                                                        {
                                                                return step;
                                                        }
                                                }
                                        }
                                }

                                // NOK, give up
                                return step0;
                        }
                };
        }
}

