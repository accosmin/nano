#include "model.h"
#include "math/numeric.h"
#include "solver_stoch.h"
#include "trainer_stoch.h"
#include "function_stoch.h"
#include "logger.h"
#include <iomanip>

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

size_t stoch_trainer_t::epoch_size(const task_t& task, const size_t fold) const
{
        const auto train_size = task.size({fold, protocol::train});
        return idiv(train_size, m_batch);
}

void stoch_trainer_t::tune(
        const enhancer_t& enhancer, const task_t& task, const size_t fold, accumulator_t& acc)
{
        // tune the hyper-parameters
        // todo: also tune the batch factor (that geometrically increases the minibatch size)

        const timer_t timer;

        trainer_result_t opt_result;

        const auto param0 = acc.params();
        const auto solver = get_stoch_solvers().get(m_solver);
        for (const auto& config : solver->configs())
        {
                // minibatch iterator
                auto iterator = iterator_t(task, {fold, protocol::train}, m_batch);

                size_t epoch = 0;
                trainer_result_t result(config);

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

                        const auto test = valid;

                        // OK, update the optimum solution
                        const auto milis = timer.milliseconds();
                        const auto xnorm = state.x.lpNorm<2>();
                        const auto gnorm = state.convergence_criteria();
                        const auto ret = result.update(state, {milis, ++epoch, xnorm, gnorm, train, valid, test}, m_patience);

                        log_info() << std::setprecision(4)
                                << "tune[" << epoch << "/" << m_tune_epochs
                                << ":train=" << train
                                << ",valid=" << valid << "|" << nano::to_string(ret)
                                << "," << config << ",batch=" << iterator.size() << ",g=" << gnorm << ",x=" << xnorm
                                << "]" << timer.elapsed() << ".";

                        return !nano::is_done(ret);
                };

                // assembly optimization function & train the model
                const auto function = stoch_function_t(acc, enhancer, task,  iterator);
                auto params = stoch_params_t{m_tune_epochs, epoch_size(task, fold), m_epsilon, fn_ulog};
                solver->minimize(params, function, param0);

                // check if an improvement
                if (result < opt_result)
                {
                        opt_result = result;
                        m_solver_config = config;
                }
        }

        log_info() << std::setprecision(4)
                << "<<< stoch-" << m_solver << ": *tune*:" << opt_result << "," << timer.elapsed() << ".";
}

trainer_result_t stoch_trainer_t::train(
        const enhancer_t& enhancer, const task_t& task, const size_t fold, accumulator_t& acc) const
{
        // minibatch iterator
        auto iterator = iterator_t(task, {fold, protocol::train}, m_batch);

        const timer_t timer;

        size_t epoch = 0;
        trainer_result_t result(m_solver_config);

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

                log_info() << std::setprecision(4)
                        << "[" << epoch << "/" << m_epochs
                        << ":train=" << train
                        << ",valid=" << valid << "|" << nano::to_string(ret)
                        << ",test=" << test
                        << "," << m_solver_config << ",batch=" << iterator.size() << ",g=" << gnorm << ",x=" << xnorm
                        << "]" << timer.elapsed() << ".";

                return !nano::is_done(ret);
        };

        // assembly optimization function & train the model
        const auto function = stoch_function_t(acc, enhancer, task,  iterator);
        const auto params = stoch_params_t{m_epochs, epoch_size(task, fold), m_epsilon, fn_ulog};
        auto solver = get_stoch_solvers().get(m_solver);
        solver->minimize(params, function, acc.params());

        log_info() << std::setprecision(4)
                << "<<< stoch-" << m_solver << ": " << result << "," << timer.elapsed() << ".";

        // OK
        return result;
}
