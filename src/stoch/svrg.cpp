#include "svrg.h"
#include "loop.h"
#include "lrate.h"
#include "text/to_params.h"

namespace nano
{
        stoch_svrg_t::stoch_svrg_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_svrg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
        {
                return stoch_tune(this, param, function, x0, make_alpha0s(), make_decays());
        }

        state_t stoch_svrg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
                const scalar_t alpha0, const scalar_t decay) const
        {
                assert(function.size() == x0.size());

                const auto config = to_params("alpha0", alpha0, "decay", decay);

                // learning rate schedule
                lrate_t lrate(alpha0, decay);

                // current state
                state_t cstate(function.size());
                cstate.stoch_update(function, x0);

                // snapshot/final state
                state_t sstate(function.size());
                sstate.update(function, x0);

                // for each epoch ...
                for (size_t e = 0; e < param.m_max_epochs; ++ e)
                {
                        // for each iteration ...
                        for (size_t i = 0; i < param.m_epoch_size && cstate; ++ i)
                        {
                                // learning rate
                                const scalar_t alpha = lrate.get();

                                // descent direction
                                function.stoch_eval(sstate.x, &cstate.d);
                                cstate.d.noalias() = - cstate.g + cstate.d - sstate.g;

                                // update solution
                                function.stoch_next();
                                cstate.stoch_update(function, alpha);
                        }

                        // check divergence
                        if (!cstate)
                        {
                                sstate.m_status = opt_status::failed;
                                break;
                        }

                        // update snapshot
                        sstate.update(function, cstate.x);

                        // check convergence (using the full gradient)
                        if (sstate.converged(param.m_epsilon))
                        {
                                sstate.m_status = opt_status::converged;
                                param.tlog(sstate, config);
                                param.ulog(sstate, config);
                                break;
                        }

                        // log the current state & check the stopping criteria
                        param.tlog(sstate, config);
                        if (!param.ulog(sstate, config))
                        {
                                sstate.m_status = opt_status::stopped;
                                break;
                        }
                }

                // OK
                return sstate;
        }
}

