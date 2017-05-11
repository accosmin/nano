#include "model.h"
#include "math/numeric.h"
#include "trainer_batch.h"
#include "function_batch.h"
#include "text/to_params.h"
#include "batch_optimizer.h"
#include "text/from_params.h"
#include "logger.h"

namespace nano
{
        batch_trainer_t::batch_trainer_t(const string_t& parameters) :
                trainer_t(to_params(parameters, "opt", "lbfgs[...]", "epochs", "1024[4,4096]",
                "policy", to_string(policy::stop_early) + "[" + concatenate(enum_values<policy>()) + "]",
                "eps", 1e-6, "patience", 32))
        {
        }

        trainer_result_t batch_trainer_t::train(
                const iterator_t& iterator, const task_t& task, const size_t fold, const size_t nthreads, const loss_t& loss,
                model_t& model) const
        {
                // parameters
                const auto epochs = clamp(from_params<size_t>(config(), "epochs"), 4, 4096);
                const auto tpolicy = from_params<policy>(config(), "policy");
                const auto epsilon = from_params<scalar_t>(config(), "eps");
                const auto optimizer = from_params<string_t>(config(), "opt");
                const auto patience = from_params<size_t>(config(), "patience");

                // acumulator
                accumulator_t acc(model, loss);
                acc.threads(nthreads);

                const timer_t timer;
                size_t epoch = 0;
                trainer_result_t result;

                // logging operator
                const auto fn_ulog = [&] (const state_t& state)
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
                        const auto ret = result.update(state, {milis, ++epoch, xnorm, gnorm, train, valid, test}, string_t{}, patience);

                        log_info()
                                << "[" << epoch << "/" << epochs
                                << ":train=" << train
                                << ",valid=" << valid << "|" << nano::to_string(ret)
                                << ",test=" << test
                                << ",g=" << gnorm << ",x=" << xnorm
                                << "] " << timer.elapsed() << ".";

                        return !nano::is_done(ret, tpolicy);
                };

                // assembly optimization function & train the model
                const auto function = batch_function_t(acc, iterator, task, fold);
                const auto params = batch_params_t{epochs, epsilon, fn_ulog};
                get_batch_optimizers().get(optimizer)->minimize(params, function, model.params());

                log_info() << "<<< batch-" << optimizer << ": " << result << ",time=" << timer.elapsed() << ".";

                // OK
                if (result.valid())
                {
                        model.params(result.optimum_params());
                }
                return result;
        }
}
