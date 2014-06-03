#include "batch_trainer.h"
#include "common/math.hpp"
#include "common/logger.h"
#include "sampler.h"
#include "accumulator.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        batch_trainer_t::batch_trainer_t(const string_t& parameters)
                :       trainer_t(parameters,
                                  "batch trainer, "\
                                  "parameters: opt=lbfgs[,cgd,gd],iters=1024[4,4096],eps=1e-6[1e-8,1e-3]")
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

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

                trainer_state_t state(model.psize());

                // train the model
                const scalars_t lambdas = { 1e-3, 1e-2, 1e-1, 1.0 };
                for (scalar_t lambda : lambdas)
                {
                        accumulator_t ldata(model, accumulator_t::type::value, lambda);
                        accumulator_t gdata(model, accumulator_t::type::vgrad, lambda);

                        const vector_t x0 = model.params();
                        ncv::batch_train(task, tsampler.get(), vsampler.get(), nthreads,
                                         loss, optimizer, iterations / 8, 8, epsilon,
                                         x0, ldata, gdata, state);
                }

                log_info() << "optimum [train = " << state.m_tvalue << "/" << state.m_terror
                           << ", valid = " << state.m_vvalue << "/" << state.m_verror
                           << ", lambda = " << state.m_lambda
                           << ", funs = " << state.m_fcalls << "/" << state.m_gcalls
                           << "].";

                // OK
                return model.load_params(state.m_params);
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
