#include "batch_trainer.h"
#include "core/logger.h"
#include "core/optimize.h"
#include "core/timer.h"
#include "core/text.h"
#include "core/clamp.hpp"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        batch_trainer_t::batch_trainer_t(const string_t& params)
                :       m_optimizer(text::from_params<string_t>(params, "opt", "lbfgs")),
                        m_iterations(text::from_params<size_t>(params, "iter", 256)),
                        m_epsilon(text::from_params<scalar_t>(params, "eps", 1e-6))
        {
                m_iterations = math::clamp(m_iterations, 4, 1024);
                m_epsilon = math::clamp(m_epsilon, 1e-8, 1e-3);
        }

        //-------------------------------------------------------------------------------------------------

        static void update(const optimize::result_t& result, timer_t& timer)
        {
                ncv::log_info() << "batch trainer: state [loss = " << result.optimum().f
                                << ", gradient = " << result.optimum().g.lpNorm<Eigen::Infinity>()
                                << ", calls = " << result.n_fval_calls() << "/" << result.n_grad_calls()
                                << "] updated in " << timer.elapsed() << ".";
                timer.start();
        }

        //-------------------------------------------------------------------------------------------------

        bool batch_trainer_t::train(const task_t& task, const fold_t& fold, const loss_t& loss, model_t& model) const
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
                samples_t samples = trainer_t::prune_annotated(task, task.samples(fold));
                if (samples.empty())
                {
                        log_error() << "batch trainer: no annotated training samples!";
                        return false;
                }

                // DEBUG!
                samples.erase(samples.begin() + 2000, samples.end());

                // optimization problem: size
                auto opt_fn_size = [&] ()
                {
                        return model.n_parameters();
                };

                // optimization problem: function value
                auto opt_fn_fval = [&] (const vector_t& x)
                {
                        model.load_params(x);
                        return trainer_t::value(task, samples, loss, model);
                };

                // optimization problem: function value & gradient
                auto opt_fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        model.load_params(x);
                        return trainer_t::vgrad(task, samples, loss, model, gx);
                };

                const optimize::problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_fval_grad);

                // optimize the model
                vector_t x(model.n_parameters());
                model.save_params(x);

                timer_t timer;

                const auto updater = std::bind(update, _1, std::ref(timer));

                optimize::result_t res;
                if (text::iequals(m_optimizer, "lbfgs"))
                {
                        res = optimize::lbfgs(problem, x, m_iterations, m_epsilon, 6, updater);
                }
                else if (text::iequals(m_optimizer, "cgd"))
                {
                        res = optimize::cgd(problem, x, m_iterations, m_epsilon, updater);
                }
                else if (text::iequals(m_optimizer, "gd"))
                {
                        res = optimize::gd(problem, x, m_iterations, m_epsilon, updater);
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
                           << "], speed = [" << res.speed().avg() << " +/- " << res.speed().stdev() << "].";

                return true;
        }

        //-------------------------------------------------------------------------------------------------
}
