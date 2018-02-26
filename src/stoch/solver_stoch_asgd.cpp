#include "lrate.h"
#include "tensor/momentum.h"
#include "solver_stoch_asgd.h"

using namespace nano;

tuner_t stoch_asgd_t::configs() const
{
        tuner_t tuner;
        tuner.add("alpha0", make_pow10_scalars(0, -3, -1)).precision(3);
        tuner.add("decay", make_scalars(0.1, 0.2, 0.5, 0.9)).precision(1);
        tuner.add("momentum", make_scalars(0.1, 0.2, 0.5, 0.9)).precision(1);
        return tuner;
}

json_reader_t& stoch_asgd_t::config(json_reader_t& reader)
{
        return reader.object("alpha0", m_alpha0, "decay", m_decay, "momentum", m_momentum);
}

json_writer_t& stoch_asgd_t::config(json_writer_t& writer) const
{
        return writer.object("alpha0", m_alpha0, "decay", m_decay, "momentum", m_momentum);
}

solver_state_t stoch_asgd_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        lrate_t lrate(m_alpha0, m_decay);

        momentum_t<vector_t> xavg(m_momentum, x0.size());

        const auto solver = [&] (solver_state_t& cstate, const solver_state_t&)
        {
                // descent direction
                xavg.update(cstate.x);
                cstate.d = -cstate.g;

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, lrate.get());
        };

        const auto snapshot = [&] (const solver_state_t&, solver_state_t& sstate)
        {
                sstate.update(function, xavg.value());
        };

        return loop(param, function, x0, solver, snapshot);
}
