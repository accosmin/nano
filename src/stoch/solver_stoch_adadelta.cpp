#include "tensor/momentum.h"
#include "solver_stoch_adadelta.h"

using namespace nano;

tuner_t stoch_adadelta_t::configs() const
{
        tuner_t tuner;
        tuner.add("momentum", make_scalars(0.1, 0.2, 0.5, 0.9)).precision(1);
        tuner.add("epsilon", make_pow10_scalars(0, -7, -2)).precision(7);
        return tuner;
}

void stoch_adadelta_t::from_json(const json_t& json)
{
        nano::from_json(json, "momentum", m_momentum, "epsilon", m_epsilon);
}

void stoch_adadelta_t::to_json(json_t& json) const
{
        nano::to_json(json, "momentum", m_momentum, "epsilon", m_epsilon);
}

solver_state_t stoch_adadelta_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        momentum_t<vector_t> gavg(m_momentum, x0.size());
        momentum_t<vector_t> davg(m_momentum, x0.size());

        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // descent direction
                gavg.update(cstate.g.array().square());

                cstate.d = -cstate.g.array() *
                           (m_epsilon + davg.value().array()).sqrt() /
                           (m_epsilon + gavg.value().array()).sqrt();

                davg.update(cstate.d.array().square());

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, 1);
        };

        const auto snapshot = [&] (const solver_state_t& cstate, solver_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return loop(param, function, x0, solver, snapshot);
}
