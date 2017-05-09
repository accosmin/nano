#include "loop.h"
#include "model.h"
#include "stochastic.h"
#include "math/numeric.h"
#include "text/to_params.h"
#include "stoch_optimizer.h"
#include "text/from_params.h"
#include "trainer_function.h"

namespace nano
{
        stochastic_trainer_t::stochastic_trainer_t(const string_t& parameters) :
                trainer_t(to_params(parameters, "opt", "sg[...]", "epochs", "16[1,1024]",
                "policy", "stop_early[,all_epochs]", "min_batch", "32[32,1024]", "max_batch", "256[32,4096]", "eps", 1e-6, "patience", 32))
        {
        }

        trainer_result_t stochastic_trainer_t::train(
                iterator_t& it_train, const iterator_t& it_valid, const iterator_t& it_test,
                const size_t nthreads, const loss_t& loss,
                model_t& model) const
        {
                // parameters
                const auto epochs = clamp(from_params<size_t>(config(), "epochs"), 1, 1024);
                const auto policy = from_params<trainer_policy>(config(), "policy");
                const auto batch0 = clamp(from_params<size_t>(config(), "min_batch"), 1, 1024);
                const auto batchK = clamp(from_params<size_t>(config(), "max_batch"), batch0, 4096);
                const auto epsilon = from_params<scalar_t>(config(), "eps");
                const auto optimizer = from_params<string_t>(config(), "opt");
                const auto patience = from_params<size_t>(config(), "patience");

                // iterator
                const auto train_size = it_train.task().size(it_train.fold());
                const auto samples = epochs * train_size;
                auto factor = scalar_t(1);
                auto epoch_size = batch0;
                if (batch0 != batchK)
                {
                        factor = clamp(scalar_t(samples - batch0) / scalar_t(samples - batchK), scalar_t(1), scalar_t(2));
                        epoch_size = idiv(static_cast<size_t>(std::log(batchK / batch0) / std::log(factor)), epochs);
                }

                it_train.reset(it_train.task(), it_train.fold(), batch0, factor);

                // accumulator
                accumulator_t acc(model, loss);
                acc.threads(nthreads);

                const timer_t timer;
                size_t epoch = 0;
                trainer_result_t result;

                // tuning operator
                const auto fn_tlog = [&] (const state_t& state, const string_t& config)
                {
                        // NB: the training state is already computed
                        const auto train = trainer_measurement_t{acc.vstats().avg(), acc.estats().avg()};

                        log_info()
                                << "[tune:train=" << train
                                << "," << config << ",batch=" << it_train.size() << ",g=" << state.convergence_criteria()
                                << "] " << timer.elapsed() << ".";

                        // NB: need to reset the minibatch size (changed during tuning)!
                        it_train.reset(it_train.task(), it_train.fold(), batch0, factor);
                };

                // logging operator
                const auto fn_ulog = [&] (const state_t& state, const string_t& config)
                {
                        return ulog(acc, it_train, it_valid, it_test, epoch, epochs, result, policy, patience, timer, state, config);
                };

                // assembly optimization function & train the model
                const auto function = trainer_function_t(acc, it_train);
                const auto params = stoch_params_t{epochs, epoch_size, epsilon, fn_ulog, fn_tlog};
                get_stoch_optimizers().get(optimizer)->minimize(params, function, model.params());

                log_info() << "<<< stoch-" << optimizer << ": " << result << ",time=" << timer.elapsed() << ".";

                // OK
                if (result.valid())
                {
                        model.params(result.optimum_params());
                }
                return result;
        }
}
