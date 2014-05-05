#include "minibatch_trainer.h"
#include "common/math.hpp"
#include "common/logger.h"
#include "sampler.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        minibatch_trainer_t::minibatch_trainer_t(const string_t& parameters)
                :       trainer_t(parameters,
                                  "minibatch trainer, parameters: batch=1024[256,8192],iters=1024[4,4096],eps=1e-6[1e-8,1e-3]")
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool minibatch_trainer_t::train(
                const task_t& task, const fold_t& fold, const loss_t& loss, size_t nthreads,
                model_t& model) const
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "minibatch trainer: cannot only train models with training samples!";
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
                const string_t optimizer = "gd";
                const size_t iterations = math::clamp(text::from_params<size_t>(configuration(), "iters", 1024), 4, 4096);
                const scalar_t epsilon = math::clamp(text::from_params<scalar_t>(configuration(), "eps", 1e-6), 1e-8, 1e-3);
                const size_t batchsize = math::clamp(text::from_params<size_t>(configuration(), "batch", 1024), 256, 8192);

                tsampler.setup(sampler_t::stype::uniform, batchsize);
                vsampler.setup(sampler_t::stype::uniform, batchsize);

                const scalars_t l2_weights = { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 };

                // L2-regularize the loss
                trainer_state_t state(model.psize());
                for (scalar_t l2_weight : l2_weights)
                {
                        trainer_t::train(task, tsampler, vsampler, nthreads,
                                         loss, l2_weight, optimizer, iterations, epsilon,
                                         model, state);
                }

                model.load_params(state.m_params);

                // OK
                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
