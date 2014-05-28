#include "minibatch_trainer.h"
#include "common/math.hpp"
#include "common/logger.h"
#include "sampler.h"
#include "accumulator.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        minibatch_trainer_t::minibatch_trainer_t(const string_t& parameters)
                :       trainer_t(parameters,
                                  "minibatch trainer, "\
                                  "parameters: opt=gd[,lbfgs,cgd],epoch=16[1,1024],"\
                                  "batch=1024[256,8192],iters=8[4,128],eps=1e-6[1e-8,1e-3]")
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
                const size_t epochs = math::clamp(text::from_params<size_t>(configuration(), "epoch", 16), 1, 1024);
                const size_t batchsize = math::clamp(text::from_params<size_t>(configuration(), "batch", 1024), 256, 8192);
                const size_t iterations = math::clamp(text::from_params<size_t>(configuration(), "iters", 8), 4, 128);
                const scalar_t epsilon = math::clamp(text::from_params<scalar_t>(configuration(), "eps", 1e-6), 1e-8, 1e-3);

                const batch_optimizer optimizer = text::from_string<batch_optimizer>
                                (text::from_params<string_t>(configuration(), "opt", "gd"));

                tsampler.setup(sampler_t::stype::uniform, batchsize);   // random training samples
                vsampler.setup(sampler_t::stype::batch);                // but all validation samples

                trainer_state_t state(model.psize());

                // train the model
                const scalars_t lambdas = { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 };
                for (scalar_t lambda : lambdas)
                {
                        accumulator_t ldata(model, accumulator_t::type::value, lambda);
                        accumulator_t gdata(model, accumulator_t::type::vgrad, lambda);

                        vector_t x0 = model.params();
                        for (size_t e = 0; e < epochs; e ++)
                        {
                                const samples_t tsamples = tsampler.get();
                                const samples_t vsamples = vsampler.get();

                                const opt_state_t result = ncv::batch_train(
                                        task, tsamples, vsamples, nthreads,
                                        loss, optimizer, 1, iterations, epsilon,
                                        x0, ldata, gdata, state);
                                x0 = result.x;
                        }
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
