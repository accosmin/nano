#include "ls_backtrack.h"

namespace nano
{
        bool ls_backtrack_t::operator()(
                const ls_strategy strategy, const scalar_t c1, const scalar_t c2,
                ls_step_t& step, const scalar_t t0,
                const scalar_t decrement,
                const scalar_t increment) const
        {
                scalar_t t = t0;
                for (int i = 0; i < 100 && t > ls_step_t::minimum() && t < ls_step_t::maximum(); ++ i)
                {
                        if (!step.update(t))
                        {
                                return false;
                        }

                        // check Armijo condition
                        if (!step.has_armijo(c1))
                        {
                                t *= decrement;
                        }
                        else if (strategy == ls_strategy::backtrack_armijo)
                        {
                                return true;
                        }

                        // check Wolfe condition
                        else if (!step.has_wolfe(c2))
                        {
                                t *= increment;
                        }
                        else if (strategy == ls_strategy::backtrack_wolfe)
                        {
                                return true;
                        }

                        // check strong Wolfe condition
                        else if (!step.has_strong_wolfe(c2))
                        {
                                t *= decrement;
                        }
                        else
                        {
                                return true;
                        }
                }

                // NOK, give up
                return false;
        }
}

