#include "stochastic_trainer.h"
#include "common/timer.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "sampler.h"
#include "accumulator.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        stochastic_trainer_t::stochastic_trainer_t(const string_t& parameters)
                :       trainer_t(parameters,
                                  "stochastic trainer, "\
                                  "parameters: opt=sg[,sga,sia],epoch=16[1,1024]")
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool stochastic_trainer_t::train(
                const task_t& task, const fold_t& fold, const loss_t& loss, size_t nthreads, model_t& model) const
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "stochastic trainer: cannot only train models with training samples!";
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

                const stochastic_optimizer optimizer = text::from_string<stochastic_optimizer>
                        (text::from_params<string_t>(configuration(), "opt", "sg"));

                trainer_state_t state(model.psize());

                // train the model
                const scalars_t lambdas = { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 };
                for (scalar_t lambda : lambdas)
                {
                        accumulator_t ldata(model, accumulator_t::type::value, lambda);
                        accumulator_t gdata(model, accumulator_t::type::vgrad, lambda);

                        const samples_t tsamples = tsampler.get();
                        const samples_t vsamples = vsampler.get();

                        const vector_t x0 = model.params();

                        ncv::stochastic_train(task, tsamples, vsamples, nthreads,
                                              loss, optimizer, epochs,
                                              x0, ldata, gdata, state);
                }

                log_info() << "[train* = " << state.m_tvalue << "/" << state.m_terror
                           << ", valid* = " << state.m_vvalue << "/" << state.m_verror << "].";

                // OK
                return model.load_params(state.m_params);
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
