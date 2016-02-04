#pragma once

#include "ls_init.hpp"
#include "batch_loop.hpp"
#include "ls_strategy.hpp"
#include <deque>

namespace math
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
                using param_t = batch_params_t<tproblem>;
                using tstate = typename param_t::tstate;
                using tscalar = typename param_t::tscalar;
                using tvector = typename param_t::tvector;
                using topulog = typename param_t::topulog;

                ///
                /// \brief constructor
                ///
                explicit batch_lbfgs_t(const param_t& param) : m_param(param)
                {
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const tproblem& problem, const tvector& x0) const
                {
                        assert(problem.size() == x0.size());

                        std::deque<tvector> ss, ys;
                        tstate istate(problem, x0);             // initial state
                        tstate pstate = istate;                 // previous state

                        tvector q, r;

                        // line-search initial step length
                        ls_init_t<tstate> ls_init(m_param.m_ls_initializer);

                        // line-search step
                        ls_strategy_t<tproblem> ls_step(m_param.m_ls_strategy, 1e-4, 0.9);

                        const auto op = [&] (tstate& cstate, const std::size_t i)
                        {
                                // descent direction
                                //      (see "Numerical optimization", Nocedal & Wright, 2nd edition, p.178)
                                q = cstate.g;

                                typename std::deque<tvector>::const_reverse_iterator itr_s = ss.rbegin();
                                typename std::deque<tvector>::const_reverse_iterator itr_y = ys.rbegin();
                                std::vector<tscalar> alphas;
                                for (std::size_t j = 1; j <= m_param.m_hsize && i >= j; j ++)
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

                                typename std::deque<tvector>::const_iterator it_s = ss.begin();
                                typename std::deque<tvector>::const_iterator it_y = ys.begin();
                                typename std::vector<tscalar>::const_reverse_iterator itr_alpha = alphas.rbegin();
                                for (std::size_t j = 1; j <= m_param.m_hsize && i >= j; j ++)
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
                                if (ss.size() > m_param.m_hsize)
                                {
                                        ss.pop_front();
                                        ys.pop_front();
                                }

                                return true;
                        };

                        // OK, assembly the optimizer
                        return batch_loop(m_param, istate, op);
                }

                // attributes
                param_t         m_param;
        };
}

