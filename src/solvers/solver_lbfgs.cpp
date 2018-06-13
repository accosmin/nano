#include "ls_init.h"
#include "ls_strategy.h"
#include "solver_lbfgs.h"
#include <deque>

using namespace nano;

void solver_lbfgs_t::from_json(const json_t& json)
{
        nano::from_json(json,
                "ls_init", m_ls_init, "ls_strat", m_ls_strat,
                "c1", m_c1, "c2", m_c2, "history", m_history_size);
}

void solver_lbfgs_t::to_json(json_t& json) const
{
        nano::to_json(json,
                "ls_init", m_ls_init, "ls_inits", join(enum_values<ls_initializer>()),
                "ls_strat", m_ls_strat, "ls_strats", join(enum_values<ls_strategy>()),
                "c1", m_c1, "c2", m_c2, "history", m_history_size);
}

solver_state_t solver_lbfgs_t::minimize(const batch_params_t& param, const function_t& function, const vector_t& x0) const
{
        // previous state
        solver_state_t pstate(function.size());

        std::deque<vector_t> ss, ys;
        vector_t q, r;

        // line-search initial step length
        ls_init_t ls_init(m_ls_init);

        // line-search step
        ls_strategy_t ls_step(m_ls_strat, m_c1, m_c2);

        const auto op = [&] (solver_state_t& cstate, const std::size_t i)
        {
                // descent direction
                //      (see "Numerical optimization", Nocedal & Wright, 2nd edition, p.178)
                q = cstate.g;

                auto itr_s = ss.rbegin();
                auto itr_y = ys.rbegin();
                std::vector<scalar_t> alphas;
                for (std::size_t j = 1; j <= m_history_size && i >= j; j ++)
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
                for (std::size_t j = 1; j <= m_history_size && i >= j; j ++)
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
                if (!ls_step(function, t0, cstate))
                {
                        return false;
                }

                // todo: may skip the update if the curvature condition is not satisfied
                // see: "A Multi-Batch L-BFGS Method for Machine Learning", page 6 - the non-convex case
                ss.emplace_back(cstate.x - pstate.x);
                ys.emplace_back(cstate.g - pstate.g);
                if (ss.size() > m_history_size)
                {
                        ss.pop_front();
                        ys.pop_front();
                }

                return true;
        };

        // assembly the solver
        return loop(param, function, x0, op);
}
