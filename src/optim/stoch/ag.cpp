#include "ag.h"
#include "lrate.h"
#include "loop.hpp"

namespace nano
{
        stoch_ag_t::stoch_ag_t(const ag_restart restart) :
                m_restart(restart)
        {
        }

        state_t stoch_ag_t::operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                const auto qs = make_finite_space(scalar_t(0.0));
                return stoch_tune(this, param, problem, x0, make_alpha0s(), make_decays(), qs);
        }

        state_t stoch_ag_t::operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                const scalar_t alpha0, const scalar_t decay, const scalar_t q) const
        {
                assert(problem.size() == x0.size());

                // initial state
                state_t istate(problem, x0);

                // current & previous iterations
                vector_t cx = istate.x;
                vector_t px = istate.x;
                vector_t cy = istate.x;
                vector_t py = istate.x;

                scalar_t cfx = istate.f;
                scalar_t pfx = istate.f;

                scalar_t ptheta = 1;
                scalar_t ctheta = 1;

                // learning rate schedule
                lrate_t lrate(alpha0, decay);

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

                const auto op_iter = [&] (state_t& cstate)
                {
                        // learning rate
                        const scalar_t alpha = lrate.get();

                        // momentum
                        ctheta = get_theta(ptheta, q);
                        const scalar_t beta = get_beta(ptheta, ctheta);

                        // update solution
                        cstate.update(problem, py);
                        cx = py - alpha * cstate.g;
                        cy = cx + beta * (cx - px);
                        cstate.x = cx; // NB: to propagate the current parameters!

                        switch (m_restart)
                        {
                        case ag_restart::none:
                                break;

                        case ag_restart::function:
                                if ((cfx = problem(cx)) > pfx)
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

                // OK, assembly the optimizer
                return  stoch_loop(problem, param, istate, op_iter,
                        {{"alpha0", alpha0}, {"decay", decay}, {"q", q}});
        }
}

