#include "cortex/model.h"
#include "math/clamp.hpp"
#include "cortex/logger.h"
#include "cortex/stochastic.h"
#include "stochastic_trainer.h"
#include "text/from_params.hpp"

namespace nano
{
        stochastic_trainer_t::stochastic_trainer_t(const string_t& parameters) :
                trainer_t(parameters)
        {
        }

        trainer_result_t stochastic_trainer_t::train(
                const task_t& task, const size_t fold, const size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                model_t& model) const
        {
                // initialize the model
                model.resize(task, true);
                model.random_params();

                // parameters
                const auto epochs = clamp(from_params<size_t>(configuration(), "epochs", 16), 1, 1024);
                const auto optimizer = from_string<stoch_optimizer>(from_params<string_t>(configuration(), "opt", "sg"));

                // train the model
                const auto result = stochastic_train(model, task, fold, nthreads, loss, criterion,
                        optimizer, epochs);

                log_info() << "<<< " << result << ".";

                // OK
                if (result.valid())
                {
                        model.load_params(result.optimum_params());
                }
                return result;
        }
}
