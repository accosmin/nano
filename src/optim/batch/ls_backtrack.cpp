#include "ls_backtrack.h"

namespace nano
{
        ls_step_t ls_backtrack_t::operator()(
                const ls_strategy strategy, const scalar_t c1, const scalar_t c2,
                const ls_step_t& step0, const scalar_t t0,
                const scalar_t decrement,
                const scalar_t increment) const
        {
                ls_step_t step(step0);

                scalar_t t = t0;
                for (int i = 0; i < 100 && t > ls_step_t::minimum() && t < ls_step_t::maximum(); ++ i)
                {
                        if (!step.update(t))
                        {
                                return step0;
                        }

                        // check Armijo condition
                        if (!step.has_armijo(c1))
                        {
                                t *= decrement;
                        }
                        else
                        {
                                if (strategy == ls_strategy::backtrack_armijo)
                                {
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
}

