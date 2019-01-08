#include "solver_nag.h"

using namespace nano;

static auto get_theta(const scalar_t ptheta, const scalar_t q)
{
        const auto a = scalar_t(1);
        const auto b = ptheta * ptheta - q;
        const auto c = - ptheta * ptheta;

        assert(b * b >= 4 * a * c);
        return (-b + std::sqrt(b * b - 4 * a * c)) / (2 * a);
}

static auto get_beta(const scalar_t ptheta, const scalar_t ctheta)
{
        return ptheta * (1 - ptheta) / (ptheta * ptheta + ctheta);
}

tuner_t solver_nag_t::tuner() const
{
        tuner_t tuner;
        tuner.add_enum<lsearch_t::initializer>("init");
        tuner.add_enum<lsearch_t::strategy>("strat");
        return tuner;
}

void solver_nag_t::from_json(const json_t& json)
{
        nano::from_json(json, "init", m_init, "strat", m_strat, "c1", m_c1, "c2", m_c2, "q", m_q);
}

void solver_nag_t::to_json(json_t& json) const
{
        nano::to_json(json,
                "init", m_init, "inits", join(enum_values<lsearch_t::initializer>()),
                "strat", m_strat, "strats", join(enum_values<lsearch_t::strategy>()),
                "c1", m_c1, "c2", m_c2, "q", m_q);
}

solver_state_t solver_nag_t::minimize(const size_t max_iterations, const scalar_t epsilon,
        const solver_function_t& function, const vector_t& x0, const logger_t& logger) const
{
        // current & previous iterations
        vector_t cx = x0;
        vector_t px = x0;
        vector_t cy = x0;
        vector_t py = x0;

        scalar_t ptheta = 1;
        scalar_t ctheta = 1;

        lsearch_t lsearch(m_init, m_strat, m_c1, m_c2);

        auto cstate = solver_state_t{function, x0};
        for (size_t i = 0; i < max_iterations; ++ i, ++ cstate.m_iterations)
        {
                // momentum
                ctheta = get_theta(ptheta, m_q);
                const auto beta = get_beta(ptheta, ctheta);

                // update solution
                cstate.update(function, py);
                cstate.d = -cstate.g;
                const auto iter_ok = lsearch(function, cstate);
                if (solver_t::done(logger, function, cstate, epsilon, iter_ok))
                {
                        break;
                }

                cx = cstate.x;
                cy = cx + beta * (cx - px);
                cstate.x = cx; // NB: to propagate the current parameters!

                if (!(cstate.g.dot(cx - px) < 0))
                {
                        ctheta = 1;
                }

                // next iteration
                px = cx;
                py = cy;
                ptheta = ctheta;
        }

        return cstate;
}
