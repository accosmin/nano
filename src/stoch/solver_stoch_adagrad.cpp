#include "text/json_writer.h"
#include "solver_stoch_adagrad.h"

using namespace nano;

solver_state_t stoch_adagrad_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        return tune(this, param, function, x0, make_alpha0s(), make_epsilons());
}

solver_state_t stoch_adagrad_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
        const scalar_t alpha0, const scalar_t epsilon)
{
        // second-order gradient momentum
        vector_t gsum2 = vector_t::Zero(x0.size());

        // assembly the solver
        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // learning rate
                const scalar_t alpha = alpha0;

                // descent direction
                gsum2.array() += cstate.g.array().square();

                cstate.d = -cstate.g.array() / (epsilon + gsum2.array().sqrt());

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, alpha);
        };

        const auto snapshot = [&] (const solver_state_t& cstate, solver_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return  loop(param, function, x0, solver, snapshot,
                json_writer_t().object("alpha0", alpha0, "epsilon", epsilon).str());
}
