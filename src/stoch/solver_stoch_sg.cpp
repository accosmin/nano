#include "lrate.h"
#include "solver_stoch_sg.h"

using namespace nano;

tuner_t stoch_sg_t::configs() const
{
        tuner_t tuner;
        tuner.add_base10("alpha0", -4, 0);
        tuner.add_linear("decay", 0.5, 1.0);
        tuner.add_base10("tnorm", 0, 2);
        return tuner;
}

json_reader_t& stoch_sg_t::config(json_reader_t& reader)
{
        return reader.object("alpha0", m_alpha0, "decay", m_decay, "tnorm", m_tnorm);
}

json_writer_t& stoch_sg_t::config(json_writer_t& writer) const
{
        return writer.object("alpha0", m_alpha0, "decay", m_decay, "tnorm", m_tnorm);
}

solver_state_t stoch_sg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        // learning rate schedule
        lrate_t lrate(m_alpha0, m_decay, m_tnorm);

        // assembly the solver
        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // learning rate
                const scalar_t alpha = lrate.get();

                // descent direction
                cstate.d = -cstate.g;

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
