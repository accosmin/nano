#include "model.h"
#include "utils.h"
#include "math/numeric.h"
#include "solver_stoch.h"
#include "trainer_stoch.h"
#include "function_stoch.h"
#include "logger.h"
#include <iomanip>

using namespace nano;

static trainer_result_t train(const task_t& task, const size_t fold, accumulator_t& acc,
        const rstoch_solver_t& solver, const string_t& config,
        const size_t epochs, const scalar_t epsilon, const size_t patience,
        const nano::timer_t& timer, const bool tuning)
{
        assert(solver);

        size_t epoch = 0;
        trainer_result_t result(config);

        // setup solver
        solver->config(config);

        // setup minibatch iterator
        scalar_t batch_factor = 1;
        json_reader_t(config).object("batchr", batch_factor);

        const size_t batch0 = physical_cpus();
        auto iterator = iterator_t(task, {fold, protocol::train}, batch0, batch_factor);

        // logging operator
        const auto fn_ulog = [&] (const solver_state_t& state)
        {
                // evaluate the current state
                // NB: the training state is already estimated!
                const auto train = measure(acc);
                const auto valid = measure(state.x, task, {fold, protocol::valid}, acc);
                const auto test  = tuning ? valid : measure(state.x, task, {fold, protocol::test}, acc);

                // OK, update the optimum solution
                const auto milis = timer.milliseconds();
                const auto xnorm = state.x.lpNorm<2>();
                const auto gnorm = state.convergence_criteria();
                const auto ret = result.update(state, {milis, ++epoch, xnorm, gnorm, train, valid, test}, patience);

                log_info() << std::setprecision(3)
                        << (tuning ? "tune[" : "[") << epoch << "/" << epochs
                        << ":train=" << train
                        << ",valid=" << valid << "|" << nano::to_string(ret)
                        << ",test=" << test
                        << "," << config << ",b=" << iterator.size() << ",g=" << gnorm << ",x=" << xnorm
                        << "]" << timer.elapsed() << ".";

                return !nano::is_done(ret);
        };

        // assembly optimization function & train the model
        const auto function = stoch_function_t(acc, task, iterator);
        const auto params = stoch_params_t{epochs, epsilon, fn_ulog};
        solver->minimize(params, function, acc.params());

        // OK
        return result;
}

json_reader_t& stoch_trainer_t::config(json_reader_t& reader)
{
        return reader.object("solver", m_solver,
                "epochs", m_epochs, "epsilon", m_epsilon, "patience", m_patience);
}

json_writer_t& stoch_trainer_t::config(json_writer_t& writer) const
{
        return writer.object("solver", m_solver, "solvers", join(get_stoch_solvers().ids()),
                "epochs", m_epochs, "epsilon", m_epsilon, "patience", m_patience);
}

void stoch_trainer_t::tune(const task_t& task, const size_t fold, accumulator_t& acc)
{
        const timer_t timer;
        trainer_result_t opt_result;

        const auto params = acc.params();
        const auto solver = get_stoch_solvers().get(m_solver);

        auto tuner = solver->configs();
        tuner.add_base10("batchr", -6, -2, 1);
        const auto trials = 10 * tuner.n_params();

        // tune the hyper-parameters: solver + minibatch increase factor
        for (size_t trial = 0; trial < trials; ++ trial)
        {
                acc.params(params);
                const auto config = tuner.get();
                const auto epochs = m_epochs;
                const auto result = ::train(task, fold, acc, solver, config, epochs, m_epsilon, m_patience, timer, true);

                // check if an improvement
                if (result < opt_result)
                {
                        opt_result = result;
                        m_tuned_config = config;
                }
        }

        assert(opt_result);
        log_info() << std::setprecision(3)
                << "<<< stoch-" << m_solver << "[tuned]: " << opt_result << "," << timer.elapsed() << ".";
}

trainer_result_t stoch_trainer_t::train(const task_t& task, const size_t fold, accumulator_t& acc) const
{
        const timer_t timer;

        const auto solver = get_stoch_solvers().get(m_solver);
        const auto config = m_tuned_config;
        const auto epochs = m_epochs;
        const auto result = ::train(task, fold, acc, solver, config, epochs, m_epsilon, m_patience, timer, false);

        assert(result);
        log_info() << std::setprecision(3)
                << "<<< stoch-" << m_solver << ": " << result << "," << timer.elapsed() << ".";
        return result;
}
