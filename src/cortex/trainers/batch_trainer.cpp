#include "cortex/batch.h"
#include "cortex/model.h"
#include "math/clamp.hpp"
#include "batch_trainer.h"
#include "cortex/logger.h"
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
                // initialize the model
                model.resize(task, true);
                model.random_params();

                // parameters
                const auto iterations = nano::clamp(nano::from_params<size_t>(configuration(), "iters", 1024), 4, 4096);
                const auto epsilon = nano::clamp(nano::from_params<scalar_t>(configuration(), "eps", 1e-6), 1e-8, 1e-3);

                const auto optimizer = nano::from_string<nano::batch_optimizer>
                        (nano::from_params<string_t>(configuration(), "opt", "lbfgs"));

                // train the model
                const auto result = nano::batch_train(model, task, fold, nthreads, loss, criterion, optimizer, iterations, epsilon);

                log_info() << "<<< " << result << ".";

                // OK
                if (result.valid())
                {
                        model.load_params(result.optimum_params());
                }
                return result;
        }
}
