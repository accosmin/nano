#include "ag.h"
#include "loop.h"
#include "text/to_params.h"

namespace nano
{
        template <ag_restart trestart>
        stoch_ag_base_t<trestart>::stoch_ag_base_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        template <ag_restart trestart>
        state_t stoch_ag_base_t<trestart>::minimize(const stoch_params_t& param,
                const function_t& function, const vector_t& x0) const
        {
                const auto qs = make_finite_space(scalar_t(0.0));
                return stoch_tune(this, param, function, x0, make_alpha0s(), qs);
        }

        template <ag_restart trestart>
        state_t stoch_ag_base_t<trestart>::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
                const scalar_t alpha0, const scalar_t q) const
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

                const auto get_theta = [] (const auto ptheta, const auto q)
                {
                        const auto a = scalar_t(1);
                        const auto b = ptheta * ptheta - q;
                        const auto c = - ptheta * ptheta;

                        return (-b + std::sqrt(b * b - 4 * a * c)) / (2 * a);
                };

                const auto get_beta = [] (const auto ptheta, const auto ctheta)
                {
                        return ptheta * (1 - ptheta) / (ptheta * ptheta + ctheta);
                };

                // assembly the optimizer
                const auto optimizer = [&] (state_t& cstate, const state_t&)
                {
                        // learning rate
                        const scalar_t alpha = alpha0;

                        // momentum
                        ctheta = get_theta(ptheta, q);
                        const scalar_t beta = get_beta(ptheta, ctheta);

                        // update solution
                        function.stoch_next();
                        cstate.stoch_update(function, py);
                        cx = py - alpha * cstate.g;
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

                const auto snapshot = [&] (const state_t& cstate, state_t& sstate)
                {
                        sstate.update(function, cstate.x);
                };

                return  stoch_loop(param, function, x0, optimizer, snapshot,
                        to_params("alpha0", alpha0, "q", q));
        }

        template struct stoch_ag_base_t<ag_restart::none>;
        template struct stoch_ag_base_t<ag_restart::function>;
        template struct stoch_ag_base_t<ag_restart::gradient>;
}

