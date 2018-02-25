#include "tensor/momentum.h"
#include "solver_stoch_adadelta.h"

using namespace nano;

tuner_t stoch_adadelta_t::configs() const
{
        tuner_t tuner;
        tuner.add("momentum", make_scalars(0.10, 0.20, 0.50, 0.90)).precision(2);
        return tuner;
}

json_reader_t& stoch_adadelta_t::config(json_reader_t& reader)
{
        return reader.object("momentum", m_momentum, "epsilon", m_epsilon);
}

json_writer_t& stoch_adadelta_t::config(json_writer_t& writer) const
{
        return writer.object("momentum", m_momentum, "epsilon", m_epsilon);
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
