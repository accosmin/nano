#include "lrate.h"
#include "solver_stoch_ngd.h"

using namespace nano;

tuner_t stoch_ngd_t::configs() const
{
        tuner_t tuner;
        tuner.add_base10("alpha0", -6, 0);
        tuner.add_linear("decay", 0, 1);
        return tuner;
}

json_reader_t& stoch_ngd_t::config(json_reader_t& reader)
{
        return reader.object("alpha0", m_alpha0, "decay", m_decay);
}

json_writer_t& stoch_ngd_t::config(json_writer_t& writer) const
{
        return writer.object("alpha0", m_alpha0, "decay", m_decay);
}

solver_state_t stoch_ngd_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        // learning rate schedule
        lrate_t lrate(m_alpha0, m_decay);

        // assembly the solver
        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // learning rate
                const scalar_t alpha = lrate.get();

                // descent direction
                const scalar_t norm = 1 / cstate.g.template lpNorm<2>();
                cstate.d = -cstate.g * norm;

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
