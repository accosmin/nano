#include "ls_backtrack.h"

namespace nano
{
        enum class ls_status
        {
                next,
                done
        };

        template <typename toperator>
        static ls_step_t ls_backtrack_loop(
                const ls_step_t& step0, const scalar_t t0, const toperator& op)
        {
                auto t = t0;
                auto step = step0;

                for (int i = 0; i < 100 && t > ls_step_t::minimum() && t < ls_step_t::maximum(); ++ i)
                {
                        if (!step.update(t))
                        {
                                return step0;
                        }

                        switch (op(step, t))
                        {
                        case ls_status::next:   continue;
                        case ls_status::done:   return step;
                        }
                }

                return step0;
        }

        ls_step_t ls_backtrack_armijo_t::operator()(
                const scalar_t c1, const scalar_t,
                const ls_step_t& step0, const scalar_t t0,
                const scalar_t decrement,
                const scalar_t) const
        {
                const auto op = [=] (ls_step_t& step, scalar_t& t)
                {
                        if (!step.has_armijo(c1))
                        {
                                t *= decrement;
                                return ls_status::next;
                        }
                        else
                        {
                                step.update(t);
                                return ls_status::done;
                        }
                };

                return ls_backtrack_loop(step0, t0, op);
        }

        ls_step_t ls_backtrack_wolfe_t::operator()(
                const scalar_t c1, const scalar_t c2,
                const ls_step_t& step0, const scalar_t t0,
                const scalar_t decrement,
                const scalar_t increment) const
        {
                const auto op = [=] (ls_step_t& step, scalar_t& t)
                {
                        if (!step.has_armijo(c1))
                        {
                                t *= decrement;
                                return ls_status::next;
                        }
                        else if (!step.has_wolfe(c2))
                        {
                                t *= increment;
                                return ls_status::next;
                        }
                        else
                        {
                                return ls_status::done;
                        }
                };

                return ls_backtrack_loop(step0, t0, op);
        }

        ls_step_t ls_backtrack_strong_wolfe_t::operator()(
                const scalar_t c1, const scalar_t c2,
                const ls_step_t& step0, const scalar_t t0,
                const scalar_t decrement,
                const scalar_t increment) const
        {
                const auto op = [=] (ls_step_t& step, scalar_t& t)
                {
                        if (!step.has_armijo(c1))
                        {
                                t *= decrement;
                                return ls_status::next;
                        }
                        else if (!step.has_wolfe(c2))
                        {
                                t *= increment;
                                return ls_status::next;
                        }
                        else if (!step.has_strong_wolfe(c2))
                        {
                                t *= decrement;
                                return ls_status::next;
                        }
                        else
                        {
                                return ls_status::done;
                        }
                };

                return ls_backtrack_loop(step0, t0, op);
        }
}

