#include "tensor/momentum.h"
#include "text/json_writer.h"
#include "solver_stoch_adadelta.h"

using namespace nano;

solver_state_t stoch_adadelta_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        return tune(this, param, function, x0, make_momenta(), make_epsilons());
}

solver_state_t stoch_adadelta_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
        const scalar_t momentum, const scalar_t epsilon)
{
        // second-order momentum of the gradient
        momentum_t<vector_t> gavg(momentum, x0.size());

        // second-order momentum of the step updates
        momentum_t<vector_t> davg(momentum, x0.size());

        // assembly the solver
        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // descent direction
                gavg.update(cstate.g.array().square());

                cstate.d = -cstate.g.array() *
                           (epsilon + davg.value().array().sqrt()) /
                           (epsilon + gavg.value().array().sqrt());

                davg.update(cstate.d.array().square());

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, 1);
        };

        const auto snapshot = [&] (const solver_state_t& cstate, solver_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return  loop(param, function, x0, solver, snapshot,
                json_writer_t().object("momentum", momentum, "epsilon", epsilon).str());
}
