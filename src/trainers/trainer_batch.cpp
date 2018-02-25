#include "model.h"
#include "utils.h"
#include "tuner.h"
#include "math/numeric.h"
#include "solver_batch.h"
#include "trainer_batch.h"
#include "function_batch.h"
#include "logger.h"
#include <iomanip>

using namespace nano;

static trainer_result_t train(const task_t& task, const size_t fold, accumulator_t& acc,
        const rbatch_solver_t& solver, const string_t& config,
        const size_t epochs, const scalar_t epsilon, const size_t patience,
        const nano::timer_t& timer)
{
        assert(solver);

        size_t epoch = 0;
        trainer_result_t result(config);

        // setup solver
        solver->config(config);

        // setup L2-regularization
        scalar_t lambda = 0;
        json_reader_t(config).object("lambda", lambda);
        acc.lambda(lambda);

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
                        << "," << config << ",g=" << gnorm << ",x=" << xnorm
                        << "]" << timer.elapsed() << ".";

                return !nano::is_done(ret);
        };

        // assembly optimization function & train the model
        const auto function = batch_function_t(acc, task, fold_t{fold, protocol::train});
        const auto params = batch_params_t{epochs, epsilon, fn_ulog};
        solver->minimize(params, function, acc.params());

        return result;
}

json_reader_t& batch_trainer_t::config(json_reader_t& reader)
{
        return reader.object("solver", m_solver, "epochs", m_epochs, "epsilon", m_epsilon, "patience", m_patience);
}

json_writer_t& batch_trainer_t::config(json_writer_t& writer) const
{
        return writer.object(
                "solver", m_solver, "solvers", join(get_batch_solvers().ids()),
                "epochs", m_epochs, "epsilon", m_epsilon, "patience", m_patience);
}

trainer_result_t batch_trainer_t::train(const task_t& task, const size_t fold, accumulator_t& acc) const
{
        const timer_t timer;
        trainer_result_t result;

        const auto params = acc.params();
        const auto solver = get_batch_solvers().get(m_solver);

        tuner_t tuner;
        tuner.add("lambda", make_pow10_scalars(0, -6, -1));

        // tune the hyper-parameters: solver + L2-regularizer
        for (const auto& config : tuner.get(10 * tuner.n_params()))
        {
                const auto epochs = m_epochs;
                const auto epsilon = m_epsilon;
                const auto patience = m_patience;

                acc.params(params);
                result = std::min(result, ::train(task, fold, acc, solver, config, epochs, epsilon, patience, timer));
        }

        assert(result);
        log_info() << std::setprecision(3) << "<<< batch-" << m_solver << ": " << result << "," << timer.elapsed() << ".";
        return result;
}
