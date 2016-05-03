#include "math/clamp.hpp"
#include "cortex/model.h"
#include "cortex/logger.h"
#include "cortex/minibatch.h"
#include "minibatch_trainer.h"
#include "text/from_params.hpp"

namespace nano
{
        minibatch_trainer_t::minibatch_trainer_t(const string_t& parameters) :
                trainer_t(parameters)
        {
        }

        trainer_result_t minibatch_trainer_t::train(
                const task_t& task, const size_t fold, const size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                model_t& model) const
        {
                // initialize the model
                model.resize(task, true);
                model.random_params();

                // parameters
                const auto epochs = nano::clamp(nano::from_params<size_t>(configuration(), "epoch", 16), 1, 1024);
                const auto epsilon = nano::clamp(nano::from_params<scalar_t>(configuration(), "eps", 1e-6), 1e-8, 1e-3);

                const auto optimizer = nano::from_string<nano::batch_optimizer>
                        (nano::from_params<string_t>(configuration(), "opt", "gd"));

                // train the model
                const auto result = minibatch_train(model, task, fold, nthreads, loss, criterion, optimizer, epochs, epsilon);

                log_info() << "<<< " << result << ".";

                // OK
                if (result.valid())
                {
                        model.load_params(result.optimum_params());
                }
                return result;
        }
}
