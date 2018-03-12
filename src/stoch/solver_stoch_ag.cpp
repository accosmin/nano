#include "solver_stoch_ag.h"

using namespace nano;

static scalar_t get_theta(const scalar_t ptheta, const scalar_t q)
{
        const auto a = scalar_t(1);
        const auto b = ptheta * ptheta - q;
        const auto c = - ptheta * ptheta;

        assert(b * b >= 4 * a * c);
        return (-b + std::sqrt(b * b - 4 * a * c)) / (2 * a);
}

static scalar_t get_beta(const scalar_t ptheta, const scalar_t ctheta)
{
        return ptheta * (1 - ptheta) / (ptheta * ptheta + ctheta);
}

template <ag_restart trestart>
tuner_t stoch_ag_base_t<trestart>::configs() const
{
        tuner_t tuner;
        tuner.add("alpha0", make_pow10_scalars(0, -3, -1)).precision(3);
        tuner.add("q", make_scalars(0.0, 0.1, 0.2, 0.5, 1.0));
        return tuner;
}

template <ag_restart trestart>
void stoch_ag_base_t<trestart>::from_json(const json_t& json)
{
        nano::from_json(json, "alpha0", m_alpha0, "q", m_q);
}

template <ag_restart trestart>
void stoch_ag_base_t<trestart>::to_json(json_t& json) const
{
        nano::to_json(json, "alpha0", m_alpha0, "q", m_q);
}

template <ag_restart trestart>
solver_state_t stoch_ag_base_t<trestart>::minimize(const stoch_params_t& param,
        const function_t& function, const vector_t& x0) const
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

        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // momentum
                ctheta = get_theta(ptheta, m_q);
                const scalar_t beta = get_beta(ptheta, ctheta);

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, py);
                cx = py - m_alpha0 * cstate.g;
                cy = cx + beta * (cx - px);
                cstate.x = cx; // NB: to propagate the current parameters!

                switch (trestart)
                {
                case ag_restart::none:
                        break;

                case ag_restart::function:
                        if ((cfx = function.stoch_eval(cx)) > pfx)
                        {
                                ctheta = 1;
                        }
                        break;

                case ag_restart::gradient:
                        if (cstate.g.dot(cx - px) > scalar_t(0))
                        {
                                ctheta = 1;
                        }
                        break;
                }

                // next iteration
                px = cx;
                py = cy;
                pfx = cfx;
                ptheta = ctheta;
        };

        const auto snapshot = [&] (const solver_state_t& cstate, solver_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return loop(param, function, x0, solver, snapshot);
}

template class nano::stoch_ag_base_t<ag_restart::none>;
template class nano::stoch_ag_base_t<ag_restart::function>;
template class nano::stoch_ag_base_t<ag_restart::gradient>;
