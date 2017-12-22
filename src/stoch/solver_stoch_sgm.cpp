#include "lrate.h"
#include "tensor/momentum.h"
#include "text/json_writer.h"
#include "solver_stoch_sgm.h"

using namespace nano;

solver_state_t stoch_sgm_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        return tune(this, param, function, x0, make_alpha0s(), make_decays(), make_momenta());
}

solver_state_t stoch_sgm_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
        const scalar_t alpha0, const scalar_t decay, const scalar_t momentum)
{
        // learning rate schedule
        lrate_t lrate(alpha0, decay, param.m_epoch_size);

        // first-order momentum of the gradient
        momentum_t<vector_t> gsum1(momentum, x0.size());

        // assembly the solver
        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // learning rate
                const scalar_t alpha = lrate.get();

                // descent direction
                gsum1.update(cstate.g);

                cstate.d = -gsum1.value();

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, alpha);
        };

        const auto snapshot = [&] (const solver_state_t& cstate, solver_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return  loop(param, function, x0, solver, snapshot,
                json_writer_t().object("alpha0", alpha0, "decay", decay, "momentum", momentum).str());
}
