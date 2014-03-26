#include "minibatch_trainer.h"
#include "common/math.hpp"
#include "common/logger.h"
#include "common/usampler.hpp"
#include "common/random.hpp"
#include "common/timer.h"
#include "common/logger.h"
#include "optimize/opt_gd.hpp"
#include "optimize/opt_cgd.hpp"
#include "optimize/opt_lbfgs.hpp"
#include "text.h"

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
                const samples_t samples = ncv::prune_annotated(task, task.samples(fold));
                if (samples.empty())
                {
                        log_error() << "minibatch trainer: no annotated training samples!";
                        return false;
                }

                samples_t tsamples, vsamples;
                ncv::uniform_split(samples, size_t(90), random_t<size_t>(0, samples.size()), tsamples, vsamples);

                samples_t utsamples, uvsamples;

                // parameters
                const size_t batchsize = math::clamp(text::from_params<size_t>(parameters(), "batch", 1024), 256, 8192);
                const size_t iterations = math::clamp(text::from_params<size_t>(parameters(), "iters", 1024), 4, 4096);
                const scalar_t epsilon = math::clamp(text::from_params<scalar_t>(parameters(), "eps", 1e-6), 1e-8, 1e-3);

                // construct the optimization problem
                const timer_t timer;

                trainer_data_t ldata(model, trainer_data_t::type::value);
                trainer_data_t gdata(model, trainer_data_t::type::vgrad);

                trainer_state_t state(model.n_parameters());

                auto fn_size = [&] ()
                {
                        return ldata.n_parameters();
                };

                auto fn_fval = [&] (const vector_t& x)
                {
                        // training samples: loss value
                        ldata.clear(x);
                        ldata.update_mt(task, utsamples, loss, nthreads);
                        const scalar_t tvalue = ldata.value();
                        const scalar_t terror = ldata.error();

                        // validation samples: loss value
                        ldata.clear(x);
                        ldata.update_mt(task, uvsamples, loss, nthreads);
                        const scalar_t vvalue = ldata.value();
                        const scalar_t verror = ldata.error();

                        // update the optimum state
                        state.update(x, tvalue, terror, vvalue, verror);

                        return tvalue;
                };

                auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        utsamples = ncv::uniform_sample(tsamples, batchsize, random_t<size_t>(0, tsamples.size()));
                        uvsamples = ncv::uniform_sample(vsamples, batchsize, random_t<size_t>(0, vsamples.size()));

                        // training samples: loss value & gradient
                        gdata.clear(x);
                        gdata.update_mt(task, utsamples, loss, nthreads);
                        const scalar_t tvalue = gdata.value();
                        const scalar_t terror = gdata.error();
                        gx = gdata.vgrad();

                        // validation samples: loss value
                        ldata.clear(x);
                        ldata.update_mt(task, uvsamples, loss, nthreads);
                        const scalar_t vvalue = ldata.value();
                        const scalar_t verror = ldata.error();

                        // update the optimum state
                        state.update(x, tvalue, terror, vvalue, verror);

                        return tvalue;
                };

                auto fn_wlog = [] (const string_t& message)
                {
                        log_warning() << message;
                };
                auto fn_elog = [] (const string_t& message)
                {
                        log_error() << message;
                };
                auto fn_ulog = [&] (const opt_state_t& result, const timer_t& timer)
                {
                        const scalar_t tvalue = state.m_tvalue;
                        const scalar_t terror = state.m_terror;
                        const scalar_t vvalue = state.m_vvalue;
                        const scalar_t verror = state.m_verror;

                        log_info() << "[loss = " << result.f
                                   << ", grad = " << result.g.lpNorm<Eigen::Infinity>()
                                   << ", funs = " << result.n_fval_calls() << "/" << result.n_grad_calls()
                                   << ", train* = " << tvalue << "/" << terror
                                   << ", valid* = " << vvalue << "/" << verror
                                   << "] done in " << timer.elapsed() << ".";
                };

                // assembly optimization problem & optimize the model
                const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

                const vector_t x = model.params();

                const opt_opulog_t fn_ulog_ref = std::bind(fn_ulog, _1, std::cref(timer));

                optimize::gd(problem, x, iterations, epsilon, fn_wlog, fn_elog, fn_ulog_ref);

                model.load_params(state.m_params);

                // OK
                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
