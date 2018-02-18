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
        const nano::timer_t& timer)
{
        assert(solver);

        size_t epoch = 0;
        trainer_result_t result(config);

        // setup solver
        solver->config(config);

        // setup minibatch iterator
        scalar_t batch_factor = 1;
        json_reader_t(config).object("batchr", batch_factor);

        const size_t batch0 = 1;
        auto iterator = iterator_t(task, {fold, protocol::train}, batch0, batch_factor);

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
                const auto ret = result.update(state, {milis, ++epoch, xnorm, gnorm, train, valid, test}, patience);

                log_info() << std::setprecision(3)
                        << "[" << epoch << "/" << epochs
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

trainer_result_t stoch_trainer_t::train(const task_t& task, const size_t fold, accumulator_t& acc) const
{
        const timer_t timer;
        trainer_result_t result;

        const auto params = acc.params();
        const auto solver = get_stoch_solvers().get(m_solver);

        auto tuner = solver->configs();
        tuner.add_finite("batchr", make_scalars(1.1, 1.2, 1.5, 2.0));
        const auto trials = 10 * tuner.n_params();

        // tune the hyper-parameters: solver + minibatch increase factor
        for (size_t trial = 0; trial < trials; ++ trial)
        {
                const auto config = tuner.get();
                const auto epochs = m_epochs;
                const auto epsilon = m_epsilon;
                const auto patience = m_patience;

                acc.params(params);
                result = std::min(result, ::train(task, fold, acc, solver, config, epochs, epsilon, patience, timer));
        }

        assert(result);
        log_info() << std::setprecision(3) << "<<< stoch-" << m_solver << ": " << result << "," << timer.elapsed() << ".";
        return result;
}
