#include "batch_trainer.h"
#include "util/math.hpp"
#include "util/logger.h"
#include "util/usampler.hpp"
#include "util/timer.h"
#include "util/logger.h"
#include "optimize/opt_gd.hpp"
#include "optimize/opt_cgd.hpp"
#include "optimize/opt_lbfgs.hpp"
#include "text.h"
#include "trainer_data.h"
#include "trainer_state.h"

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
                const task_t& task, const samples_t& tsamples, const samples_t& vsamples, const loss_t& loss,
                const model_t& model, size_t nthreads, scalar_t l2reg, trainer_state_t& state) const
        {
                // construct the optimization problem
                timer_t timer;

                trainer_data_skipgrad_t ldata(model);
                trainer_data_withgrad_t gdata(model);

                auto fn_size = [&] ()
                {
                        return ldata.n_parameters();
                };

                auto fn_fval = [&] (const vector_t& x)
                {
                        // training samples: loss value
                        ldata.load_params(x);
                        ldata.update_mt(task, tsamples, loss, nthreads);
                        const scalar_t tvalue = ldata.value() + l2reg * x.squaredNorm();
                        const scalar_t terror = ldata.error();

                        // validation samples: loss value
                        ldata.load_params(x);
                        ldata.update_mt(task, vsamples, loss, nthreads);
                        const scalar_t vvalue = ldata.value() + l2reg * x.squaredNorm();;
                        const scalar_t verror = ldata.error();

                        // update the optimum state
                        state.update(x, tvalue, terror, vvalue, verror);

                        if (terror < 2.0 * verror)
                        {
                                throw std::runtime_error("over-fitting detected!");
                        }

                        return tvalue;
                };

                auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        // training samples: loss value & gradient
                        gdata.load_params(x);
                        gdata.update_mt(task, tsamples, loss, nthreads);
                        const scalar_t tvalue = ldata.value() + l2reg * x.squaredNorm();
                        const scalar_t terror = gdata.error();
                        gx = gdata.vgrad() + 2.0 * l2reg * x;

                        // validation samples: loss value
                        ldata.load_params(x);
                        ldata.update_mt(task, vsamples, loss, nthreads);
                        const scalar_t vvalue = ldata.value() + l2reg * x.squaredNorm();;
                        const scalar_t verror = ldata.error();

                        // update the optimum state
                        state.update(x, tvalue, terror, vvalue, verror);

                        if (terror < 2.0 * verror)
                        {
                                throw std::runtime_error("over-fitting detected!");
                        }

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
                auto fn_ulog = [&] (const opt_state_t& result, timer_t& timer)
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
                                   << ", L2-reg = " << l2reg
                                   << "] done in " << timer.elapsed() << ".";
                        timer.start();
                };

                // assembly optimization problem & optimize the model
                const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

                const vector_t x = model.params();

                const opt_opulog_t fn_ulog_ref = std::bind(fn_ulog, _1, std::ref(timer));

                if (text::iequals(m_optimizer, "lbfgs"))
                {
                        optimize::lbfgs(problem, x, m_iterations, m_epsilon, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else if (text::iequals(m_optimizer, "cgd"))
                {
                        optimize::cgd(problem, x, m_iterations, m_epsilon, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else if (text::iequals(m_optimizer, "gd"))
                {
                        optimize::gd(problem, x, m_iterations, m_epsilon, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else
                {
                        log_error() << "batch trainer: invalid optimization method <" << m_optimizer << ">!";
                        return false;
                }

                // OK
                return true;
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
                ncv::uniform_split(samples, size_t(90), random_t<size_t>(0, samples.size()), tsamples, vsamples);

                // tune the L2-norm regularization
                const scalars_t l2regs = { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e+0 };

                trainer_state_t state(model.n_parameters());
                for (scalar_t l2reg : l2regs)
                {
                        try
                        {
                                if (!train(task, tsamples, vsamples, loss, model, nthreads, l2reg, state))
                                {
                                        return false;
                                }
                        }
                        catch (std::exception&)
                        {
                                // over-fitting detected!
                        }
                }

                model.load_params(state.m_params);

                // OK
                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
