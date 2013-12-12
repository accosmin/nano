#include "batch_trainer.h"
#include "core/timer.h"
#include "core/text.h"
#include "core/math/clamp.hpp"
#include "core/logger.h"

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

                // prune training data
                const samples_t samples = ncv::prune_annotated(task, task.samples(fold));
                if (samples.empty())
                {
                        log_error() << "batch trainer: no annotated training samples!";
                        return false;
                }

                // optimization problem: size
                auto fn_size = [&] ()
                {
                        return model.n_parameters();
                };

                // optimization problem: function value
                auto fn_fval = [&] (const vector_t& x)
                {
                        model.load_params(x);
                        return ncv::lvalue_mt(task, samples, loss, nthreads, model);
                };

                // optimization problem: function value & gradient
                auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        model.load_params(x);
                        return ncv::lvgrad_mt(task, samples, loss, nthreads, model, gx);
                };

                // optimization problem: logging
                auto fn_wlog = [] (const string_t& message)
                {
                        log_warning() << message;
                };
                auto fn_elog = [] (const string_t& message)
                {
                        log_error() << message;
                };
                auto fn_ulog = [] (const opt_result_t& result, timer_t& timer)
                {
                        log_info() << "batch trainer: state [loss = " << result.optimum().f
                                   << ", gradient = " << result.optimum().g.lpNorm<Eigen::Infinity>()
                                   << ", calls = " << result.n_fval_calls() << " fun/" << result.n_grad_calls()
                                   << " grad] updated in " << timer.elapsed() << ".";
                        timer.start();
                };

                // assembly optimization problem
                const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);
                opt_result_t res;

                timer_t timer;

                // optimize the model
                vector_t x(model.n_parameters());
                model.save_params(x);

                const auto fn_ulog_ref = std::bind(fn_ulog, _1, std::ref(timer));
                const scalar_t eps = m_epsilon;
                const size_t iters = m_iterations;

                if (text::iequals(m_optimizer, "lbfgs"))
                {
                        res = optimizer_t::lbfgs(problem, x, iters, eps, 6, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else if (text::iequals(m_optimizer, "cgd"))
                {
                        res = optimizer_t::cgd(problem, x, iters, eps, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else if (text::iequals(m_optimizer, "gd"))
                {
                        res = optimizer_t::gd(problem, x, iters, eps, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else
                {
                        log_error() << "batch trainer: invalid optimization method <" << m_optimizer << ">!";
                        return false;
                }

                model.load_params(res.optimum().x);

                // OK
                log_info() << "batch trainer: optimum [loss = " << res.optimum().f
                           << ", gradient = " << res.optimum().g.norm()
                           << ", calls = " << res.n_fval_calls() << "/" << res.n_grad_calls()
                           << "], iterations = [" << res.iterations() << "/" << m_iterations
                           << "].";

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
