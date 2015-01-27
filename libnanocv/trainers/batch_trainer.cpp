#include "batch_trainer.h"
#include "libnanocv/util/logger.h"
#include "libnanocv/util/math.hpp"
#include "batch.h"
#include "libnanocv/model.h"
#include "libnanocv/sampler.h"

namespace ncv
{
        batch_trainer_t::batch_trainer_t(const string_t& parameters)
                :       trainer_t(parameters)
        {
        }

        trainer_result_t batch_trainer_t::train(
                const task_t& task, const fold_t& fold, const loss_t& loss, size_t nthreads, const string_t& criterion,
                model_t& model) const
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "batch trainer: can only train models with training samples!";
                        return trainer_result_t();
                }

                // initialize the model
                model.resize(task, true);
                model.random_params();

                // prune training & validation data
                sampler_t tsampler(task);
                tsampler.setup(fold).setup(sampler_t::atype::annotated);

                sampler_t vsampler(task);
                tsampler.split(80, vsampler);

                if (tsampler.empty() || vsampler.empty())
                {
                        log_error() << "batch trainer: no annotated training samples!";
                        return trainer_result_t();
                }

                // parameters
                const size_t iterations = math::clamp(text::from_params<size_t>(configuration(), "iters", 1024), 4, 4096);
                const scalar_t epsilon = math::clamp(text::from_params<scalar_t>(configuration(), "eps", 1e-4), 1e-8, 1e-3);

                const batch_optimizer optimizer = text::from_string<batch_optimizer>
                        (text::from_params<string_t>(configuration(), "opt", "lbfgs"));

                tsampler.setup(sampler_t::stype::batch);                // all training samples
                vsampler.setup(sampler_t::stype::batch);                // all validation samples

                // train the model
                const trainer_result_t result = ncv::batch_train(
                        model, task, tsampler, vsampler, nthreads,
                        loss, criterion, optimizer, iterations, epsilon);

                log_info() << "optimum [train = " << result.m_opt_state.m_tvalue << "/" << result.m_opt_state.m_terror_avg
                           << ", valid = " << result.m_opt_state.m_vvalue << "/" << result.m_opt_state.m_verror_avg
                           << ", epoch = " << result.m_opt_epoch
                           << ", config = " << text::concatenate(result.m_opt_config, "/")
                           << "].";

                // OK
                if (result.valid())
                {
                        model.load_params(result.m_opt_params);
                }
                return result;
        }
}
