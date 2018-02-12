#include "model.h"
#include "utils.h"
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
                "batch", m_batch, "batch_ratio", m_batch_ratio,
                "epsilon", m_epsilon, "patience", m_patience);
}

json_writer_t& stoch_trainer_t::config(json_writer_t& writer) const
{
        return writer.object("solver", m_solver, "solvers", join(get_stoch_solvers().ids()),
                "tune_epochs", m_tune_epochs, "epochs", m_epochs,
                "batch", m_batch, "batch_ratio", m_batch_ratio,
                "epsilon", m_epsilon, "patience", m_patience);
}

size_t stoch_trainer_t::epoch_size(const task_t& task, const size_t fold) const
{
        const auto train_size = task.size({fold, protocol::train});
        return idiv(train_size, m_batch);
}

void stoch_trainer_t::tune(const task_t& task, const size_t fold, accumulator_t& acc)
{
        const auto solver = get_stoch_solvers().get(m_solver);
        if (!solver)
        {
                assert(solver);
                log_error() << "unknown solver [" << m_solver << "]!";
                return;
        }

        const timer_t timer;
        trainer_result_t opt_result;
        const auto param0 = acc.params();

        auto tuner = solver->configs();
        const auto trials = 10 * tuner.n_params();

        // tune the hyper-parameters
        // todo: also tune the batch factor (that geometrically increases the minibatch size)
        // todo: also tune the L2-regularization term
        for (size_t trial = 0; trial < trials; ++ trial)
        {
                const auto config = tuner.get();

                // minibatch iterator
                auto iterator = iterator_t(task, {fold, protocol::train}, m_batch, m_batch_ratio);

                size_t epoch = 0;
                trainer_result_t result(config);

                // logging operator
                const auto fn_ulog = [&] (const solver_state_t& state)
                {
                        // evaluate the current state
                        // NB: the training state is already estimated!
                        const auto train = measure(acc);
                        const auto valid = measure(state.x, task, {fold, protocol::valid}, acc);
                        const auto test  = valid;

                        // OK, update the optimum solution
                        const auto milis = timer.milliseconds();
                        const auto xnorm = state.x.lpNorm<2>();
                        const auto gnorm = state.convergence_criteria();
                        const auto ret = result.update(state, {milis, ++epoch, xnorm, gnorm, train, valid, test}, m_patience);

                        log_info() << std::setprecision(3)
                                << "tune[" << epoch << "/" << m_tune_epochs
                                << ":train=" << train
                                << ",valid=" << valid << "|" << nano::to_string(ret)
                                << "," << config << ",batch=" << iterator.size() << ",g=" << gnorm << ",x=" << xnorm
                                << "]" << timer.elapsed() << ".";

                        return !nano::is_done(ret);
                };

                // assembly optimization function & train the model
                const auto function = stoch_function_t(acc, task,  iterator);
                const auto params = stoch_params_t{m_tune_epochs, epoch_size(task, fold), m_epsilon, fn_ulog};
                solver->minimize(params, function, param0);

                // check if an improvement
                if (result < opt_result)
                {
                        opt_result = result;
                        m_solver_config = config;
                }
        }

        assert(opt_result);
        log_info() << std::setprecision(3)
                << "<<< stoch-" << m_solver << "[tuned]: " << opt_result << "," << timer.elapsed() << ".";
}

trainer_result_t stoch_trainer_t::train(const task_t& task, const size_t fold, accumulator_t& acc) const
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
                const auto train = measure(acc);
                const auto valid = measure(state.x, task, {fold, protocol::valid}, acc);
                const auto test  = measure(state.x, task, {fold, protocol::test}, acc);

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
                        << "," << m_solver_config << ",batch=" << iterator.size() << ",g=" << gnorm << ",x=" << xnorm
                        << "]" << timer.elapsed() << ".";

                return !nano::is_done(ret);
        };

        // assembly optimization function & train the model
        const auto function = stoch_function_t(acc, task,  iterator);
        const auto params = stoch_params_t{m_epochs, epoch_size(task, fold), m_epsilon, fn_ulog};
        const auto solver = get_stoch_solvers().get(m_solver);
        if (!solver)
        {
                assert(solver);
                log_error() << "unknown solver [" << m_solver << "]!";
        }
        else
        {
                solver->minimize(params, function, acc.params());
        }

        assert(result);
        log_info() << std::setprecision(3) << "<<< stoch-" << m_solver << ": " << result << "," << timer.elapsed() << ".";
        return result;
}
