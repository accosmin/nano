#include "batch.h"
#include "loop.hpp"
#include "model.h"
#include "math/clamp.hpp"
#include "math/epsilon.hpp"
#include "batch_optimizer.h"
#include "text/to_string.hpp"
#include "text/from_params.hpp"

namespace nano
{
        batch_trainer_t::batch_trainer_t(const string_t& parameters) :
                trainer_t(parameters)
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

                const auto optimizer = from_params<string_t>(config(), "opt");

                // train the model
                const timer_t timer;
                const auto op = [&] (const accumulator_t& lacc, const accumulator_t& gacc, const vector_t& x0)
                {
                        auto batch_optimizer = get_batch_optimizers().get(optimizer);
                        return train(task, fold, lacc, gacc, x0, *batch_optimizer);
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

        trainer_result_t batch_trainer_t::train(
                const task_t& task, const size_t fold,
                const accumulator_t& lacc, const accumulator_t& gacc, const vector_t& x0,
                const batch_optimizer_t& optimizer) const
        {
                const timer_t timer;

                // parameters
                const auto epochs = clamp(from_params<size_t>(config(), "epochs"), 4, 4096);
                const auto policy = from_params<trainer_policy>(config(), "policy");
                const auto epsilon = epsilon0<scalar_t>();

                const auto train_fold = fold_t{fold, protocol::train};
                size_t epoch = 0;
                trainer_result_t result;

                task_iterator_t it(task, train_fold);

                // logging operator
                auto fn_ulog = make_trainer_logger(lacc, it, epoch, epochs, result, policy, timer);

                // assembly optimization problem & optimize the model
                const auto problem = make_trainer_problem(lacc, gacc, it);
                const auto params = batch_params_t{epochs, epsilon, fn_ulog};
                optimizer.minimize(params, problem, x0);

                return result;
        }
}
