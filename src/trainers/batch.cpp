#include "loop.h"
#include "batch.h"
#include "model.h"
#include "math/numeric.h"
#include "text/to_params.h"
#include "batch_optimizer.h"
#include "text/from_params.h"
#include "trainer_function.h"

namespace nano
{
        batch_trainer_t::batch_trainer_t(const string_t& parameters) :
                trainer_t(to_params(parameters, "opt", "lbfgs[...]", "epochs", "1024[4,4096]",
                "policy", "stop_early[,all_epochs]", "eps", 1e-6, "patience", 32))
        {
        }

        trainer_result_t batch_trainer_t::train(
                const task_t& task, const size_t fold, const size_t nthreads, const loss_t& loss,
                model_t& model) const
        {
                // parameters
                const auto epochs = clamp(from_params<size_t>(config(), "epochs"), 4, 4096);
                const auto policy = from_params<trainer_policy>(config(), "policy");
                const auto epsilon = from_params<scalar_t>(config(), "eps");
                const auto optimizer = from_params<string_t>(config(), "opt");
                const auto patience = from_params<size_t>(config(), "patience");

                const auto it_train = make_iterator("default", "", task, fold_t{fold, protocol::train});
                const auto it_valid = make_iterator("default", "", task, fold_t{fold, protocol::valid});
                const auto it_test = make_iterator("default", "", task, fold_t{fold, protocol::test});

                // acumulator
                accumulator_t acc(model, loss);
                acc.threads(nthreads);

                const timer_t timer;
                size_t epoch = 0;
                trainer_result_t result;

                // logging operator
                const auto fn_ulog = [&] (const state_t& state)
                {
                        return ulog(acc, it_train, it_valid, it_test, epoch, epochs, result, policy, patience, timer, state);
                };

                // assembly optimization function & train the model
                const auto function = trainer_function_t(acc, it_train);
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
