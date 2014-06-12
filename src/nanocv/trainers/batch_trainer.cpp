#include "batch_trainer.h"
#include "common/math.hpp"
#include "common/logger.h"
#include "sampler.h"
#include "accumulator.h"

namespace ncv
{
        batch_trainer_t::batch_trainer_t(const string_t& parameters)
                :       trainer_t(parameters,
                                  "batch trainer, "\
                                  "parameters: opt=lbfgs[,cgd,gd],iters=1024[4,4096],eps=1e-6[1e-8,1e-3]")
        {
        }

        bool batch_trainer_t::train(
                const task_t& task, const fold_t& fold, const loss_t& loss, size_t nthreads,
                model_t& model) const
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "batch trainer: can only train models with training samples!";
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
                        log_error() << "batch trainer: no annotated training samples!";
                        return false;
                }

                // parameters
                const size_t iterations = math::clamp(text::from_params<size_t>(configuration(), "iters", 1024), 4, 4096);
                const scalar_t epsilon = math::clamp(text::from_params<scalar_t>(configuration(), "eps", 1e-6), 1e-8, 1e-3);

                const batch_optimizer optimizer = text::from_string<batch_optimizer>
                                (text::from_params<string_t>(configuration(), "opt", "lbfgs"));

                // train the model
                trainer_result_t result(model.psize(), iterations / 8);
                ncv::batch_train(task, tsampler, vsampler, nthreads, 
                                 loss, optimizer, 1, iterations / 8, 8, epsilon,
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
