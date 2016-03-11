#pragma once

#include "ls_init.hpp"
#include "batch_loop.hpp"
#include "ls_strategy.hpp"
#include <deque>

namespace nano
{
        ///
        /// \brief limited memory bfgs (l-bfgs)
        ///
        template
        <
                typename tproblem                       ///< optimization problem
        >
        struct batch_lbfgs_t
        {
                using tparam = batch_params_t<tproblem>;
                using tstate = typename tparam::tstate;
                using tscalar = typename tparam::tscalar;
                using tvector = typename tparam::tvector;

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const tparam& param, const tproblem& problem, const tvector& x0) const
                {
                        assert(problem.size() == x0.size());

                        std::deque<tvector> ss, ys;
                        tstate istate(problem, x0);             // initial state
                        tstate pstate = istate;                 // previous state

                        tvector q, r;

                        // line-search initial step length
                        ls_init_t<tstate> ls_init(param.m_ls_initializer);

                        // line-search step
                        ls_strategy_t<tproblem> ls_step(param.m_ls_strategy, 1e-4, 0.9);

                        const auto op = [&] (tstate& cstate, const std::size_t i)
                        {
                                // descent direction
                                //      (see "Numerical optimization", Nocedal & Wright, 2nd edition, p.178)
                                q = cstate.g;

                                auto itr_s = ss.rbegin();
                                auto itr_y = ys.rbegin();
                                std::vector<tscalar> alphas;
                                for (std::size_t j = 1; j <= param.m_hsize && i >= j; j ++)
                                {
                                        const tvector& s = (*itr_s ++);
                                        const tvector& y = (*itr_y ++);

                                        const tscalar alpha = s.dot(q) / s.dot(y);
                                        q.noalias() -= alpha * y;
                                        alphas.push_back(alpha);
                                }

                                if (i == 0)
                                {
                                        r = q;
                                }
                                else
                                {
                                        const tvector& s = *ss.rbegin();
                                        const tvector& y = *ys.rbegin();
                                        r = s.dot(y) / y.dot(y) * q;
                                }

                                auto it_s = ss.begin();
                                auto it_y = ys.begin();
                                auto itr_alpha = alphas.rbegin();
                                for (std::size_t j = 1; j <= param.m_hsize && i >= j; j ++)
                                {
                                        const tvector& s = (*it_s ++);
                                        const tvector& y = (*it_y ++);

                                        const tscalar alpha = *(itr_alpha ++);
                                        const tscalar beta = y.dot(r) / s.dot(y);
                                        r.noalias() += s * (alpha - beta);
                                }

                                cstate.d = -r;

                                // line-search
                                pstate = cstate;

                                const tscalar t0 = ls_init(cstate);
                                if (!ls_step(problem, t0, cstate))
                                {
                                        return false;
                                }

                                ss.push_back(cstate.x - pstate.x);
                                ys.push_back(cstate.g - pstate.g);
                                if (ss.size() > param.m_hsize)
                                {
                                        ss.pop_front();
                                        ys.pop_front();
                                }

                                return true;
                        };

                        // OK, assembly the optimizer
                        return batch_loop(param, istate, op);
                }
        };
}

