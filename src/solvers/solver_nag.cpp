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
        tuner.add("alpha0", make_pow10_scalars(0, -3, -1)).precision(3);
        tuner.add("q", make_scalars(0.0, 0.1, 0.2, 0.5, 1.0));
        return tuner;
}

template <nag_restart trestart>
void solver_nag_base_t<trestart>::from_json(const json_t& json)
{
        nano::from_json(json, "alpha0", m_alpha0, "q", m_q);
}

template <nag_restart trestart>
void solver_nag_base_t<trestart>::to_json(json_t& json) const
{
        nano::to_json(json, "alpha0", m_alpha0, "q", m_q);
}

template <nag_restart trestart>
solver_state_t solver_nag_base_t<trestart>::minimize(const batch_params_t& param,
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

        const auto op = [&] (solver_state_t& cstate, const size_t)
        {
                // momentum
                ctheta = get_theta(ptheta, m_q);
                const auto beta = get_beta(ptheta, ctheta);

                // update solution
                cstate.update(function, py);
                cx = py - m_alpha0 * cstate.g;
                cy = cx + beta * (cx - px);
                cstate.x = cx; // NB: to propagate the current parameters!

                switch (trestart)
                {
                case nag_restart::function:
                        if ((cfx = function.eval(cx)) > pfx)
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

        return loop(param, function, x0, op);
}

template class nano::solver_nag_base_t<nag_restart::none>;
template class nano::solver_nag_base_t<nag_restart::function>;
template class nano::solver_nag_base_t<nag_restart::gradient>;
