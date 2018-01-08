#include "solver_stoch_adagrad.h"

using namespace nano;

strings_t stoch_adagrad_t::configs() const
{
        return {};
}

json_reader_t& stoch_adagrad_t::config(json_reader_t& reader)
{
        return reader.object("alpha0", m_alpha0, "epsilon", m_epsilon);
}

json_writer_t& stoch_adagrad_t::config(json_writer_t& writer) const
{
        return writer.object("alpha0", m_alpha0, "epsilon", m_epsilon);
}

solver_state_t stoch_adagrad_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        // second-order gradient momentum
        vector_t gsum2 = vector_t::Zero(x0.size());

        // assembly the solver
        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // learning rate
                const scalar_t alpha = m_alpha0;

                // descent direction
                gsum2.array() += cstate.g.array().square();

                cstate.d = -cstate.g.array() / (m_epsilon + gsum2.array().sqrt());

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
