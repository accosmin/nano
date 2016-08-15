#include "ls_backtrack.h"

namespace nano
{
        ls_step_t ls_backtrack_armijo_t::operator()(
                const scalar_t c1, const scalar_t,
                const ls_step_t& step0, const scalar_t t0,
                const scalar_t decrement,
                const scalar_t) const
        {
                ls_step_t step(step0);

                scalar_t t = t0;
                for (int i = 0; i < 100 && t > ls_step_t::minimum() && t < ls_step_t::maximum(); ++ i)
                {
                        if (!step.update(t))
                        {
                                return step0;
                        }

                        if (!step.has_armijo(c1))
                        {
                                t *= decrement;
                        }
                        else
                        {
                                return step;
                        }
                }

                // NOK, give up
                return step0;
        }

        ls_step_t ls_backtrack_wolfe_t::operator()(
                const scalar_t c1, const scalar_t c2,
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

                        if (!step.has_armijo(c1))
                        {
                                t *= decrement;
                        }
                        else if (!step.has_wolfe(c2))
                        {
                                t *= increment;
                        }
                        else
                        {
                                return step;
                        }
                }

                // NOK, give up
                return step0;
        }

        ls_step_t ls_backtrack_strong_wolfe_t::operator()(
                const scalar_t c1, const scalar_t c2,
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

                        if (!step.has_armijo(c1))
                        {
                                t *= decrement;
                        }
                        else if (!step.has_wolfe(c2))
                        {
                                t *= increment;
                        }
                        else if (!step.has_strong_wolfe(c2))
                        {
                                t *= decrement;
                        }
                        else
                        {
                                return step;
                        }
                }

                // NOK, give up
                return step0;
        }
}

