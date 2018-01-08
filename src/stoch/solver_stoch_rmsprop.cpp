#include "lrate.h"
#include "tensor/momentum.h"
#include "solver_stoch_rmsprop.h"

using namespace nano;

strings_t stoch_rmsprop_t::configs() const
{
        // todo
        return {};
}

json_reader_t& stoch_rmsprop_t::config(json_reader_t& reader)
{
        return reader.object("alpha0", m_alpha0, "decay", m_decay, "momentum", m_momentum, "epsilon", m_epsilon);
}

json_writer_t& stoch_rmsprop_t::config(json_writer_t& writer) const
{
        return writer.object("alpha0", m_alpha0, "decay", m_decay, "momentum", m_momentum, "epsilon", m_epsilon);
}

solver_state_t stoch_rmsprop_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        // learning rate schedule
        lrate_t lrate(m_alpha0, m_decay);

        // second-order momentum of the gradient
        momentum_t<vector_t> gsum2(m_momentum, x0.size());

        // assembly the solver
        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // learning rate
                const scalar_t alpha = lrate.get();

                // descent direction
                gsum2.update(cstate.g.array().square());

                cstate.d = -cstate.g.array() / (m_epsilon + gsum2.value().array()).sqrt();

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, alpha);
        };

        const auto snapshot = [&] (const solver_state_t& cstate, solver_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return loop(param, function, x0, solver, snapshot);
}
