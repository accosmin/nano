#include "stochastic_trainer.h"
#include "common/timer.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "sampler.h"
#include "accumulator.h"

namespace ncv
{
        stochastic_trainer_t::stochastic_trainer_t(const string_t& parameters)
                :       trainer_t(parameters,
                                  "stochastic trainer, "\
                                  "parameters: opt=sg[,sga,sia],epoch=16[1,1024]")
        {
        }

        bool stochastic_trainer_t::train(
                const task_t& task, const fold_t& fold, const loss_t& loss, size_t nthreads, model_t& model) const
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "stochastic trainer: can only train models with training samples!";
                        return false;
                }

                // initialize the model
                model.resize(task, true);
                model.random_params();

                // prune training & validation data
                sampler_t tsampler(task);
                tsampler.setup(fold).setup(sampler_t::atype::annotated);

                sampler_t vsampler(task);
                tsampler.split(90, vsampler);

                if (tsampler.empty() || vsampler.empty())
                {
                        log_error() << "stochastic trainer: no annotated training samples!";
                        return false;
                }

                // parameters
                const size_t epochs = math::clamp(text::from_params<size_t>(configuration(), "epoch", 16), 1, 1024);

                const stochastic_optimizer optimizer = text::from_string<stochastic_optimizer>
                        (text::from_params<string_t>(configuration(), "opt", "sg"));

                // train the model
                trainer_result_t result(model.psize(), epochs);
                ncv::stochastic_train(task, tsampler, vsampler, nthreads,
                                      loss, optimizer, epochs,
                                      model, result);

                log_info() << "optimum [train = " << result.m_opt_state.m_tvalue << "/" << result.m_opt_state.m_terror
                           << ", valid = " << result.m_opt_state.m_vvalue << "/" << result.m_opt_state.m_verror
                           << ", epoch = " << result.m_opt_epoch << "/" << result.m_epochs
                           << ", config = " << text::concatenate(result.m_opt_config, "/")
                           << "].";

                // OK
                return model.load_params(result.m_opt_params);
        }
}
