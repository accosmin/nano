#include "model.h"
#include "utils.h"
#include "tuner.h"
#include "solver.h"
#include "math/numeric.h"
#include "trainer_batch.h"
#include "function_batch.h"
#include "logger.h"
#include <iomanip>

using namespace nano;

static trainer_result_t train(const task_t& task, const size_t fold, accumulator_t& acc,
        const rsolver_t& solver,
        const size_t epochs, const scalar_t epsilon, const size_t patience,
        const nano::timer_t& timer)
{
        assert(solver);

        size_t epoch = 0;
        trainer_result_t result("");

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
                        << ",g=" << gnorm << ",x=" << xnorm
                        << "]" << timer.elapsed() << ".";

                return !nano::is_done(ret);
        };

        // assembly optimization function & train the model
        const auto function = batch_function_t(acc, task, fold_t{fold, protocol::train});
        solver->minimize(epochs, epsilon, function, acc.params(), fn_ulog);

        return result;
}

void batch_trainer_t::from_json(const json_t& json)
{
        nano::from_json(json, "solver", m_solver, "epochs", m_epochs, "epsilon", m_epsilon, "patience", m_patience);
}

void batch_trainer_t::to_json(json_t& json) const
{
        nano::to_json(json,
                "solver", m_solver, "solvers", join(get_solvers().ids()),
                "epochs", m_epochs, "epsilon", m_epsilon, "patience", m_patience);
}

trainer_result_t batch_trainer_t::train(const task_t& task, const size_t fold, accumulator_t& acc) const
{
        const timer_t timer;

        // todo: tune regulizer (L1, L2, ...)
        const auto solver = get_solvers().get(m_solver);
        const auto result = ::train(task, fold, acc, solver, m_epochs, m_epsilon, m_patience, timer);

        assert(result);
        log_info() << std::setprecision(3) << "<<< " << m_solver << ": " << result << "," << timer.elapsed() << ".";
        return result;
}
