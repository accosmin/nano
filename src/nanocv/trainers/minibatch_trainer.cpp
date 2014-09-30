#include "minibatch_trainer.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "batch.h"
#include "model.h"
#include "sampler.h"
#include "loss.h"

namespace ncv
{
        minibatch_trainer_t::minibatch_trainer_t(const string_t& parameters)
                :       trainer_t(parameters,
                                  "parameters: opt=gd[,lbfgs,cgd],epoch=16[1,1024],"\
                                  "batch=1024[32,8192],iters=8[4,128],eps=1e-6[1e-8,1e-3]")
        {
        }

        trainer_result_t minibatch_trainer_t::train(
                const task_t& task, const fold_t& fold, const loss_t& loss, size_t nthreads, const string_t& criterion,
                model_t& model) const
        {                
                if (fold.second != protocol::train)
                {
                        log_error() << "minibatch trainer: can only train models with training samples!";
                        return trainer_result_t();
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
                        log_error() << "minibatch trainer: no annotated training samples!";
                        return trainer_result_t();
                }

                // parameters
                const size_t epochs = math::clamp(text::from_params<size_t>(configuration(), "epoch", 16), 1, 1024);
                const size_t batchsize = math::clamp(text::from_params<size_t>(configuration(), "batch", 1024), 32, 8192);
                const size_t iterations = math::clamp(text::from_params<size_t>(configuration(), "iters", 8), 4, 128);
                const scalar_t epsilon = math::clamp(text::from_params<scalar_t>(configuration(), "eps", 1e-6), 1e-8, 1e-3);

                const batch_optimizer optimizer = text::from_string<batch_optimizer>
                                (text::from_params<string_t>(configuration(), "opt", "gd"));

                tsampler.setup(sampler_t::stype::uniform, batchsize);   // random training samples
                vsampler.setup(sampler_t::stype::batch);                // but all validation samples

                // train the model
                const trainer_result_t result = ncv::batch_train(
                        model, task, tsampler, vsampler, nthreads,
                        loss, criterion, optimizer, epochs, 1, iterations, epsilon);

                log_info() << "optimum [train = " << result.m_opt_state.m_tvalue << "/" << result.m_opt_state.m_terror
                           << ", valid = " << result.m_opt_state.m_vvalue << "/" << result.m_opt_state.m_verror
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
