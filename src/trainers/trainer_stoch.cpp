#include "model.h"
#include "math/numeric.h"
#include "trainer_stoch.h"
#include "function_stoch.h"
#include "solver_stoch.h"
#include "logger.h"

namespace nano
{
        stoch_trainer_t::stoch_trainer_t(const string_t& parameters) :
                trainer_t(to_params(parameters, "solver", "sg[" + concatenate(get_stoch_solvers().ids()) + "]", "epochs", "16[1,1024]",
                "min_batch", "32[32,1024]", "max_batch", "256[32,4096]", "eps", 1e-6, "patience", 32))
        {
        }

        trainer_result_t stoch_trainer_t::train(
                const iterator_t& iterator, const task_t& task, const size_t fold, const size_t nthreads, const loss_t& loss,
                model_t& model) const
        {
                // parameters
                const auto epochs = clamp(from_params<size_t>(config(), "epochs"), 1, 1024);
                const auto batch0 = clamp(from_params<size_t>(config(), "min_batch"), 1, 1024);
                const auto batchK = clamp(from_params<size_t>(config(), "max_batch"), batch0, 4096);
                const auto epsilon = from_params<scalar_t>(config(), "eps");
                const auto solver = from_params<string_t>(config(), "solver");
                const auto patience = from_params<size_t>(config(), "patience");

                // minibatch
                const auto train_size = task.size({fold, protocol::train});
                const auto samples = epochs * train_size;
                auto factor = scalar_t(1);
                auto epoch_size = batch0;
                if (batch0 != batchK)
                {
                        factor = clamp(scalar_t(samples - batch0) / scalar_t(samples - batchK), scalar_t(1), scalar_t(2));
                        epoch_size = idiv(static_cast<size_t>(std::log(batchK / batch0) / std::log(factor)), epochs);
                }

                auto minibatch = minibatch_t(task, {fold, protocol::train}, batch0, factor);

                // accumulator
                accumulator_t acc(model, loss);
                acc.threads(nthreads);

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
                                << "," << config << ",batch=" << minibatch.size() << ",g=" << gnorm << ",x=" << xnorm
                                << "] " << timer.elapsed() << ".";

                        // NB: need to reset the minibatch size (changed during tuning)!
                        minibatch.reset(batch0, factor);
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
                                << "," << config << ",batch=" << minibatch.size() << ",g=" << gnorm << ",x=" << xnorm
                                << "] " << timer.elapsed() << ".";

                        return !nano::is_done(ret);
                };

                // assembly optimization function & train the model
                const auto function = stoch_function_t(acc, iterator, task,  minibatch);
                const auto params = stoch_params_t{epochs, epoch_size, epsilon, fn_ulog, fn_tlog};
                get_stoch_solvers().get(solver)->minimize(params, function, model.params());

                log_info() << "<<< stoch-" << solver << ": " << result << ",time=" << timer.elapsed() << ".";

                // OK
                if (result.valid())
                {
                        model.params(result.optimum_params());
                }
                return result;
        }
}
