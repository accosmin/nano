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

template <nag_restart trestart>
tuner_t solver_nag_base_t<trestart>::tuner() const
{
        tuner_t tuner;
        tuner.add_enum<lsearch_t::initializer>("init");
        tuner.add_enum<lsearch_t::strategy>("strat");
        return tuner;
}

template <nag_restart trestart>
void solver_nag_base_t<trestart>::from_json(const json_t& json)
{
        nano::from_json(json, "init", m_init, "strat", m_strat, "c1", m_c1, "c2", m_c2, "q", m_q);
}

template <nag_restart trestart>
void solver_nag_base_t<trestart>::to_json(json_t& json) const
{
        nano::to_json(json,
                "init", m_init, "inits", join(enum_values<lsearch_t::initializer>()),
                "strat", m_strat, "strats", join(enum_values<lsearch_t::strategy>()),
                "c1", m_c1, "c2", m_c2, "q", m_q);
}

template <nag_restart trestart>
solver_state_t solver_nag_base_t<trestart>::minimize(const size_t max_iterations, const scalar_t epsilon,
        const function_t& f, const vector_t& x0, const logger_t& logger) const
{
        // current & previous iterations
        vector_t cx = x0;
        vector_t px = x0;
        vector_t cy = x0;
        vector_t py = x0;

        scalar_t cfx = 0;
        scalar_t pfx = std::numeric_limits<scalar_t>::max();

        scalar_t ptheta = 1;
        scalar_t ctheta = 1;

        lsearch_t lsearch(m_init, m_strat, m_c1, m_c2);

        const auto op = [&] (const function_t& function, solver_state_t& cstate, const size_t)
        {
                // momentum
                ctheta = get_theta(ptheta, m_q);
                const auto beta = get_beta(ptheta, ctheta);

                // update solution
                cstate.update(function, py);
                cstate.d = -cstate.g;
                if (!lsearch(function, cstate))
                {
                        return false;
                }
                cx = cstate.x;
                //cx = py - m_alpha0 * cstate.g;
                cy = cx + beta * (cx - px);
                cstate.x = cx; // NB: to propagate the current parameters!

                switch (trestart)
                {
                case nag_restart::function:
                        if ((cfx = function.vgrad(cx)) > pfx)
                        {
                                ctheta = 1;
                        }
                        break;

                case nag_restart::gradient:
                        if (cstate.g.dot(cx - px) > scalar_t(0))
                        {
                                ctheta = 1;
                        }
                        break;

                default:
                        break;
                }

                // next iteration
                px = cx;
                py = cy;
                pfx = cfx;
                ptheta = ctheta;

                return true;
        };

        return loop(f, x0, max_iterations, epsilon, logger, op);
}

template class nano::solver_nag_base_t<nag_restart::none>;
template class nano::solver_nag_base_t<nag_restart::function>;
template class nano::solver_nag_base_t<nag_restart::gradient>;
