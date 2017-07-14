#include "model.h"
#include "math/numeric.h"
#include "trainer_stoch.h"
#include "function_stoch.h"
#include "solver_stoch.h"
#include "logger.h"

using namespace nano;

stoch_trainer_t::stoch_trainer_t(const string_t& params) :
        trainer_t(to_params(params, "solver", "sg[" + concatenate(get_stoch_solvers().ids()) + "]",
        "epochs", "16[1,1024]", "batch", "32[32,1024]", "eps", 1e-6, "patience", 32))
{
}

trainer_result_t stoch_trainer_t::train(
        const enhancer_t& enhancer, const task_t& task, const size_t fold, accumulator_t& acc) const
{
        // parameters
        const auto epochs = clamp(from_params<size_t>(config(), "epochs"), 1, 1024);
        const auto batch0 = clamp(from_params<size_t>(config(), "batch"), 1, 1024);
        const auto solver = from_params<string_t>(config(), "solver");
        const auto epsilon = from_params<scalar_t>(config(), "eps");
        const auto patience = from_params<size_t>(config(), "patience");

        // minibatch iterator
        const auto train_size = task.size({fold, protocol::train});
        const auto epoch_size = idiv(train_size, batch0);

        auto iterator = iterator_t(task, {fold, protocol::train}, batch0);

        log_info() << "setup:epochs=" << epochs << ",epoch_size=" << epoch_size << ",batch0=" << batch0;

        const timer_t timer;
        size_t epoch = 0;
        trainer_result_t result;

        // tuning operator
        const auto fn_tlog = [&] (const function_state_t& state, const string_t& config)
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
                iterator.reset(batch0);
        };

        // logging operator
        const auto fn_ulog = [&] (const function_state_t& state, const string_t& config)
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
                const auto ret = result.update(state, {milis, ++epoch, xnorm, gnorm, train, valid, test}, config, patience);

                log_info()
                        << "[" << epoch << "/" << epochs
                        << ":train=" << train
                        << ",valid=" << valid << "|" << nano::to_string(ret)
                        << ",test=" << test
                        << "," << config << ",batch=" << iterator.size() << ",g=" << gnorm << ",x=" << xnorm
                        << "] " << timer.elapsed() << ".";

                return !nano::is_done(ret);
        };

        // assembly optimization function & train the model
        const auto function = stoch_function_t(acc, enhancer, task,  iterator);
        auto params = stoch_params_t{epochs, epoch_size, epsilon, fn_ulog, fn_tlog};
        params.m_tune_max_epochs = 1;
        params.m_tune_epoch_size = std::max(epoch_size / 10, size_t(10));
        get_stoch_solvers().get(solver)->minimize(params, function, acc.params());

        log_info() << "<<< stoch-" << solver << ": " << result << ",time=" << timer.elapsed() << ".";

        // OK
        return result;
}
