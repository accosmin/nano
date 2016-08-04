#include "lbfgs.h"
#include "loop.hpp"
#include "ls_init.h"
#include "ls_strategy.h"
#include "text/from_params.hpp"
#include <deque>

namespace nano
{
        batch_lbfgs_t::batch_lbfgs_t(const string_t& configuration) :
                batch_optimizer_t(configuration)
        {
        }

        state_t batch_lbfgs_t::minimize(const batch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                assert(problem.size() == x0.size());

                std::deque<vector_t> ss, ys;
                state_t istate(problem, x0);             // initial state
                state_t pstate = istate;                 // previous state

                vector_t q, r;

                // line-search initial step length
                ls_init_t ls_init(from_params<ls_initializer>(configuration(), "ls_init", ls_initializer::quadratic));

                // line-search step
                ls_strategy_t ls_step(from_params<ls_strategy>(configuration(), "ls_strat", ls_strategy::interpolation),
                        scalar_t(1e-4), scalar_t(0.9));

                const auto op = [&] (state_t& cstate, const std::size_t i)
                {
                        // descent direction
                        //      (see "Numerical optimization", Nocedal & Wright, 2nd edition, p.178)
                        q = cstate.g;

                        auto itr_s = ss.rbegin();
                        auto itr_y = ys.rbegin();
                        std::vector<scalar_t> alphas;
                        for (std::size_t j = 1; j <= param.m_lbfgs_hsize && i >= j; j ++)
                        {
                                const vector_t& s = (*itr_s ++);
                                const vector_t& y = (*itr_y ++);

                                const scalar_t alpha = s.dot(q) / s.dot(y);
                                q.noalias() -= alpha * y;
                                alphas.push_back(alpha);
                        }

                        if (i == 0)
                        {
                                r = q;
                        }
                        else
                        {
                                const vector_t& s = *ss.rbegin();
                                const vector_t& y = *ys.rbegin();
                                r = s.dot(y) / y.dot(y) * q;
                        }

                        auto it_s = ss.begin();
                        auto it_y = ys.begin();
                        auto itr_alpha = alphas.rbegin();
                        for (std::size_t j = 1; j <= param.m_lbfgs_hsize && i >= j; j ++)
                        {
                                const vector_t& s = (*it_s ++);
                                const vector_t& y = (*it_y ++);

                                const scalar_t alpha = *(itr_alpha ++);
                                const scalar_t beta = y.dot(r) / s.dot(y);
                                r.noalias() += s * (alpha - beta);
                        }

                        cstate.d = -r;

                        // line-search
                        pstate = cstate;

                        const scalar_t t0 = ls_init(cstate);
                        if (!ls_step(problem, t0, cstate))
                        {
                                return false;
                        }

                        ss.push_back(cstate.x - pstate.x);
                        ys.push_back(cstate.g - pstate.g);
                        if (ss.size() > param.m_lbfgs_hsize)
                        {
                                ss.pop_front();
                                ys.pop_front();
                        }

                        return true;
                };

                // OK, assembly the optimizer
                return batch_loop(param, istate, op);
        }
}

