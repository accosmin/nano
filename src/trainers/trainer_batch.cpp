#include "model.h"
#include "math/numeric.h"
#include "trainer_batch.h"
#include "function_batch.h"
#include "solver_batch.h"
#include "logger.h"
#include <iomanip>

using namespace nano;

json_reader_t& batch_trainer_t::config(json_reader_t& reader)
{
        return reader.object("solver", m_solver, "epochs", m_epochs, "eps", m_epsilon, "patience", m_patience);
}

json_writer_t& batch_trainer_t::config(json_writer_t& writer) const
{
        return writer.object(
                "solver", m_solver, "solvers", join(get_batch_solvers().ids()),
                "epochs", m_epochs, "eps", m_epsilon, "patience", m_patience);
}

void batch_trainer_t::tune(
        const enhancer_t&, const task_t&, const size_t, accumulator_t&)
{
        const auto solver = get_batch_solvers().get(m_solver);
        if (!solver)
        {
                assert(solver);
                log_error() << "unknown solver [" << m_solver << "]!";
        }
}

trainer_result_t batch_trainer_t::train(
        const enhancer_t& enhancer, const task_t& task, const size_t fold, accumulator_t& acc) const
{
        const timer_t timer;
        size_t epoch = 0;
        trainer_result_t result;

        // logging operator
        const auto fn_ulog = [&] (const solver_state_t& state)
        {
                // evaluate the current state
                // NB: the training state is already estimated!
                const auto train = trainer_measurement_t{acc.vstats().avg(), acc.estats().avg()};

                acc.params(state.x);
                acc.mode(accumulator_t::type::value);
                acc.update(task, {fold, protocol::valid});
                const auto valid = trainer_measurement_t{acc.vstats().avg(), acc.estats().avg()};

                acc.params(state.x);
                acc.mode(accumulator_t::type::value);
                acc.update(task, {fold, protocol::test});
                const auto test = trainer_measurement_t{acc.vstats().avg(), acc.estats().avg()};

                // OK, update the optimum solution
                const auto milis = timer.milliseconds();
                const auto xnorm = state.x.lpNorm<2>();
                const auto gnorm = state.convergence_criteria();
                const auto ret = result.update(state, {milis, ++epoch, xnorm, gnorm, train, valid, test}, m_patience);

                log_info() << std::setprecision(3)
                        << "[" << epoch << "/" << m_epochs
                        << ":train=" << train
                        << ",valid=" << valid << "|" << nano::to_string(ret)
                        << ",test=" << test
                        << ",g=" << gnorm << ",x=" << xnorm
                        << "]" << timer.elapsed() << ".";

                return !nano::is_done(ret);
        };

        // assembly optimization function & train the model
        const auto function = batch_function_t(acc, enhancer, task, fold_t{fold, protocol::train});
        const auto params = batch_params_t{m_epochs, m_epsilon, fn_ulog};
        const auto solver = get_batch_solvers().get(m_solver);
        if (!solver)
        {
                assert(solver);
                log_error() << "unknown solver [" << m_solver << "]!";
        }
        else
        {
                solver->minimize(params, function, acc.params());
        }

        log_info() << std::setprecision(3) << "<<< batch-" << m_solver << ": " << result << "," << timer.elapsed() << ".";
        return result;
}
