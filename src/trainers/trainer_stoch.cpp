#include "model.h"
#include "math/numeric.h"
#include "trainer_stoch.h"
#include "function_stoch.h"
#include "solver_stoch.h"
#include "logger.h"

using namespace nano;

json_reader_t& stoch_trainer_t::config(json_reader_t& reader)
{
        return reader.object("solver", m_solver,
                "tune_epochs", m_tune_epochs, "epochs", m_epochs,
                "batch", m_batch, "eps", m_epsilon, "patience", m_patience);
}

json_writer_t& stoch_trainer_t::config(json_writer_t& writer) const
{
        return writer.object(
                "solver", m_solver, "solvers", join(get_stoch_solvers().ids()),
                "tune_epochs", m_tune_epochs, "epochs", m_epochs,
                "batch", m_batch, "eps", m_epsilon, "patience", m_patience);
}

void stoch_trainer_t::tune(
        const enhancer_t& enhancer, const task_t& task, const size_t fold, accumulator_t& acc)
{
        // tune the hyper-parameters
        // todo: tune the batch factor (that geometrically increases the minibatch size)
}

trainer_result_t stoch_trainer_t::train(
        const enhancer_t& enhancer, const task_t& task, const size_t fold, accumulator_t& acc) const
{
        // minibatch iterator
        const auto train_size = task.size({fold, protocol::train});
        const auto epoch_size = idiv(train_size, m_batch);

        auto iterator = iterator_t(task, {fold, protocol::train}, m_batch);

        log_info() << "setup:epochs=" << m_epochs << ",epoch_size=" << epoch_size << ",batch0=" << m_batch;

        const timer_t timer;
        size_t epoch = 0;
        trainer_result_t result;


/*        // tuning operator
        const auto fn_tlog = [&] (const solver_state_t& state, const string_t& config)
        {
                // NB: the training state is already computed
                const auto train = trainer_measurement_t{acc.vstats().avg(), acc.estats().avg()};

                // OK, log the current state
                const auto xnorm = state.x.lpNorm<2>();
                const auto gnorm = state.convergence_criteria();

                log_info()
                        << "[tune:train=" << train
                        << "," << config << ",batch=" << iterator.size() << ",g=" << gnorm << ",x=" << xnorm
                        << "] " << timer.elapsed() << ".";

                // NB: need to reset the minibatch size (changed during tuning)!
                iterator.reset(m_batch);
        };*/

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

                log_info()
                        << "[" << epoch << "/" << m_epochs
                        << ":train=" << train
                        << ",valid=" << valid << "|" << nano::to_string(ret)
                        << ",test=" << test
                        //<< "," << config << ",batch=" << iterator.size() << ",g=" << gnorm << ",x=" << xnorm
                        << "] " << timer.elapsed() << ".";

                return !nano::is_done(ret);
        };

        // assembly optimization function & train the model
        const auto function = stoch_function_t(acc, enhancer, task,  iterator);
        auto params = stoch_params_t{m_epochs, epoch_size, m_epsilon, fn_ulog};
        get_stoch_solvers().get(m_solver)->minimize(params, function, acc.params());

        log_info() << "<<< stoch-" << m_solver << ": " << result << ",time=" << timer.elapsed() << ".";

        // OK
        return result;
}
