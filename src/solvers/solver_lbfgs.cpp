#include "solver_lbfgs.h"
#include <deque>

using namespace nano;

tuner_t solver_lbfgs_t::tuner() const
{
        tuner_t tuner;
        tuner.add_enum<lsearch_t::initializer>("init");
        tuner.add_enum<lsearch_t::strategy>("strat");
        return tuner;
}

void solver_lbfgs_t::from_json(const json_t& json)
{
        nano::from_json(json,
                "init", m_init, "strat", m_strat,
                "c1", m_c1, "c2", m_c2, "history", m_history_size);
}

void solver_lbfgs_t::to_json(json_t& json) const
{
        nano::to_json(json,
                "init", m_init, "inits", join(enum_values<lsearch_t::initializer>()),
                "strat", m_strat, "strats", join(enum_values<lsearch_t::strategy>()),
                "c1", m_c1, "c2", m_c2, "history", m_history_size);
}

solver_state_t solver_lbfgs_t::minimize(const size_t max_iterations, const scalar_t epsilon,
        const function_t& function, const vector_t& x0, const logger_t& logger) const
{
        lsearch_t lsearch(m_init, m_strat, m_c1, m_c2);

        // previous state
        solver_state_t pstate(function.size());

        std::deque<vector_t> ss, ys;
        vector_t q, r;

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
                if (!lsearch(function, cstate))
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
        return loop(function, x0, max_iterations, epsilon, logger, op);
}
