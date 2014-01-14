#include "batch_trainer.h"
#include "util/math.hpp"
#include "util/logger.h"
#include "util/usampler.hpp"
#include "text.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        batch_trainer_t::batch_trainer_t(const string_t& params)
                :       m_optimizer(text::from_params<string_t>(params, "opt", "lbfgs")),
                        m_iterations(text::from_params<size_t>(params, "iters", 1024)),
                        m_epsilon(text::from_params<scalar_t>(params, "eps", 1e-6))
        {
                m_iterations = math::clamp(m_iterations, 4, 4096);
                m_epsilon = math::clamp(m_epsilon, 1e-8, 1e-3);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool batch_trainer_t::train(
                const task_t& task, const fold_t& fold, const loss_t& loss, size_t nthreads,
                model_t& model) const
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "batch trainer: cannot only train models with training samples!";
                        return false;
                }

                // initialize the model
                model.resize(task);
                model.random_params();

                // prune training & validation data
                const samples_t samples = ncv::prune_annotated(task, task.samples(fold));
                if (samples.empty())
                {
                        log_error() << "batch trainer: no annotated training samples!";
                        return false;
                }

                samples_t tsamples, vsamples;
                ncv::uniform_split(samples, size_t(80), random_t<size_t>(0, samples.size()), tsamples, vsamples);

                // batch optimization
                trainer_state_t state(model.n_parameters());
                return ncv::train(task, tsamples, vsamples, loss,
                           m_optimizer, m_epsilon, m_iterations, nthreads,
                           model, state, true);
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
