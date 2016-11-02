#include "loop.h"
#include "batch.h"
#include "model.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "text/to_params.h"
#include "batch_optimizer.h"
#include "text/from_params.h"

namespace nano
{
        batch_trainer_t::batch_trainer_t(const string_t& parameters) :
                trainer_t(concat_params(parameters,
                "opt=lbfgs[...],epochs=1024[4,4096],policy=stop_early[,all_epochs]"))
        {
        }

        trainer_result_t batch_trainer_t::train(
                const task_t& task, const size_t fold, const size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                model_t& model) const
        {
                if (model != task)
                {
                        throw std::runtime_error("batch trainer: mis-matching model and task");
                }

                // parameters
                const auto epochs = clamp(from_params<size_t>(config(), "epochs"), 4, 4096);
                const auto policy = from_params<trainer_policy>(config(), "policy");
                const auto epsilon = epsilon0<scalar_t>();
                const auto optimizer = from_params<string_t>(config(), "opt");

                const auto train_fold = fold_t{fold, protocol::train};

                // train the model
                const timer_t timer;
                const auto op = [&] (const accumulator_t& lacc, const accumulator_t& gacc, const vector_t& x0)
                {
                        size_t epoch = 0;
                        trainer_result_t result;

                        task_iterator_t it(task, train_fold);

                        // logging operator
                        const auto fn_ulog = [&] (const state_t& state)
                        {
                                return ulog(lacc, it, epoch, epochs, result, policy, timer, state);
                        };

                        // assembly optimization problem & optimize the model
                        const auto problem = make_trainer_problem(lacc, gacc, it);
                        const auto params = batch_params_t{epochs, epsilon, fn_ulog};

                        get_batch_optimizers().get(optimizer)->minimize(params, problem, x0);

                        return result;
                };

                const auto result = trainer_loop(model, nthreads, loss, criterion, op);
                log_info() << "<<< batch-" << optimizer << ": " << result << ",time=" << timer.elapsed() << ".";

                // OK
                if (result.valid())
                {
                        model.load_params(result.optimum_params());
                }
                return result;
        }
}
