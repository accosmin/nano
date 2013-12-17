#include "trainer.h"
#include "core/math/math.hpp"
#include "core/thread/thread_loop.hpp"
#include "core/logger.h"
#include "core/timer.h"
#include "core/text.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace impl
        {
                static scalar_t lvalue(
                        const task_t& task, const sample_t& sample, const loss_t& loss, const model_t& model)
                {
                        const image_t& image = task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);
                        assert(image.has_target(target));

                        const vector_t output = model.value(image, sample.m_region);
                        return loss.value(target, output);
                }

                static scalar_t lvgrad(
                        const task_t& task, const sample_t& sample, const loss_t& loss, const model_t& model)
                {
                        const image_t& image = task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);
                        assert(image.has_target(target));

                        const vector_t output = model.value(image, sample.m_region);
                        model.cumulate_grad(loss.vgrad(target, output));
                        return loss.value(target, output);
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        struct value_data_t
        {
                // constructor
                value_data_t() : m_value(0.0), m_count(0)
                {
                }

                // update the cumulated loss value with a new sample
                void update(const task_t& task, const sample_t& sample, const loss_t& loss)
                {
                        update(task, sample, loss, *m_model);
                }
                void update(const task_t& task, const sample_t& sample, const loss_t& loss, const model_t& model)
                {
                        m_value += impl::lvalue(task, sample, loss, model);
                        m_count ++;
                }

                // cumulate loss value
                value_data_t& operator+=(const value_data_t& other)
                {
                        m_value += other.m_value;
                        m_count += other.m_count;
                        return *this;
                }

                scalar_t value() const { return m_value / ((m_count == 0) ? 1.0 : m_count); }

                // attributes
                scalar_t        m_value;
                size_t          m_count;
                rmodel_t        m_model;
        };

        /////////////////////////////////////////////////////////////////////////////////////////

        struct vgrad_data_t
        {
                // constructor
                vgrad_data_t(size_t n_parameters = 0) : m_value(0.0), m_count(0)
                {
                        resize(n_parameters);
                }

                void resize(size_t n_parameters)
                {
                        m_vgrad.resize(n_parameters);
                        m_vgrad.setZero();
                }

                // update the cumulated loss value & gradient with a new sample
                void update(const task_t& task, const sample_t& sample, const loss_t& loss)
                {
                        update(task, sample, loss, *m_model);
                }
                void update(const task_t& task, const sample_t& sample, const loss_t& loss, const model_t& model)
                {
                        m_value += impl::lvgrad(task, sample, loss, model);
                        m_count ++;
                }
                void store() const
                {
                        m_vgrad = m_model->grad();
                }
                void store(const model_t& model) const
                {
                        m_vgrad = model.grad();
                }

                // cumulate loss value & gradient
                vgrad_data_t& operator+=(const vgrad_data_t& other)
                {
                        m_value += other.m_value;
                        m_vgrad += other.m_vgrad;
                        m_count += other.m_count;
                        return *this;
                }

                scalar_t value() const { return m_value / ((m_count == 0) ? 1.0 : m_count); }
                vector_t vgrad() const { return m_vgrad / ((m_count == 0) ? 1.0 : m_count); }

                // attributes
                scalar_t                m_value;
                mutable vector_t        m_vgrad;
                size_t                  m_count;
                rmodel_t                m_model;
        };

        /////////////////////////////////////////////////////////////////////////////////////////

        samples_t prune_annotated(const task_t& task, const samples_t& samples)
        {
                samples_t pruned_samples;

                // keep only the samples having targets associated
                for (const sample_t& sample : samples)
                {
                        const image_t& image = task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);
                        if (image.has_target(target))
                        {
                                pruned_samples.push_back(sample);
                        }
                }

                return pruned_samples;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void split_train_valid(const samples_t& samples, size_t vpercentage, samples_t& tsamples, samples_t& vsamples)
        {
                vpercentage = math::clamp(vpercentage, size_t(1), size_t(99));

                tsamples.clear();
                vsamples.clear();

                random_t<size_t> rnd(0, 100);
                for (size_t i = 0; i < samples.size(); i ++)
                {
                        (rnd() < vpercentage ? vsamples : tsamples).push_back(samples[i]);
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t lvalue(const task_t& task, const sample_t& sample, const loss_t& loss, const model_t& model)
        {
                return impl::lvalue(task, sample, loss, model);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t lvgrad(const task_t& task, const sample_t& sample, const loss_t& loss, const model_t& model)
        {
                return impl::lvgrad(task, sample, loss, model);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t lvalue_st(const task_t& task, const samples_t& samples, const loss_t& loss, const model_t& model)
        {
                value_data_t cum_data;

                for (size_t i = 0; i < samples.size(); i ++)
                {
                        cum_data.update(task, samples[i], loss, model);
                }

                return cum_data.value();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t lvalue_mt(
                const task_t& task, const samples_t& samples, const loss_t& loss, size_t nthreads, const model_t& model)
        {
                value_data_t cum_data;

                thread_loop_cumulate<value_data_t>
                (
                        samples.size(),
                        [&] (value_data_t& data)
                        {
                                data.m_model = model.clone();
                        },
                        [&] (size_t i, value_data_t& data)
                        {
                                data.update(task, samples[i], loss);
                        },
                        [&] (const value_data_t& data)
                        {
                                cum_data += data;
                        },
                        nthreads
                );

                return cum_data.value();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t lvgrad_st(const task_t& task, const samples_t& samples, const loss_t& loss, const model_t& model,
                vector_t& lgrad)
        {
                vgrad_data_t cum_data(model.n_parameters());

                model.zero_grad();

                for (size_t i = 0; i < samples.size(); i ++)
                {
                        cum_data.update(task, samples[i], loss, model);
                }

                cum_data.store(model);

                lgrad = cum_data.vgrad();
                return cum_data.value();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t lvgrad_mt(
                const task_t& task, const samples_t& samples, const loss_t& loss, size_t nthreads,
                const model_t& model, vector_t& lgrad)
        {
                vgrad_data_t cum_data(model.n_parameters());

                thread_loop_cumulate<vgrad_data_t>
                (
                        samples.size(),
                        [&] (vgrad_data_t& data)
                        {
                                data.resize(model.n_parameters());
                                data.m_model = model.clone();
                                data.m_model->zero_grad();
                        },
                        [&] (size_t i, vgrad_data_t& data)
                        {
                                data.update(task, samples[i], loss);
                        },
                        [&] (const vgrad_data_t& data)
                        {
                                data.store();
                                cum_data += data;
                        },
                        nthreads
                );

                lgrad = cum_data.vgrad();
                return cum_data.value();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

//        bool trainer_t::optimize(
//                const task_t& task, const samples_t& tsamples, const samples_t& vsamples, const loss_t& loss,
//                const string_t& optimizer, scalar_t epsilon, size_t iterations, size_t nthreads,
//                model_t& model) const
//        {
//                vector_t optx(model.n_parameters());
//                model.save_params(optx);

//                // optimization problem: size
//                auto fn_size = [&] ()
//                {
//                        return model.n_parameters();
//                };

//                // optimization problem: function value
//                auto fn_fval = [&] (const vector_t& x)
//                {
//                        model.load_params(x);
//                        return ncv::lvalue_mt(task, tsamples, loss, nthreads, model);
//                };

//                // optimization problem: function value & gradient
//                auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
//                {
//                        model.load_params(x);
//                        return ncv::lvgrad_mt(task, tsamples, loss, nthreads, model, gx);
//                };

//                // optimization problem: logging
//                auto fn_wlog = [] (const string_t& message)
//                {
//                        log_warning() << message;
//                };
//                auto fn_elog = [] (const string_t& message)
//                {
//                        log_error() << message;
//                };
//                auto fn_ulog = [] (const opt_result_t& result, timer_t& timer)
//                {
//                        log_info() << "trainer: state [loss = " << result.optimum().f
//                                   << ", gradient = " << result.optimum().g.lpNorm<Eigen::Infinity>()
//                                   << ", calls = " << result.n_fval_calls() << " fun/" << result.n_grad_calls()
//                                   << " grad] updated in " << timer.elapsed() << ".";
//                        timer.start();
//                };

//                // assembly optimization problem
//                const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);
//                opt_result_t res;

//                timer_t timer;

//                // optimize the model
//                vector_t x(model.n_parameters());
//                model.save_params(x);

//                const auto fn_ulog_ref = std::bind(fn_ulog, _1, std::ref(timer));

//                if (text::iequals(optimizer, "lbfgs"))
//                {
//                        res = optimize::lbfgs(problem, x, iterations, epsilon, 6, fn_wlog, fn_elog, fn_ulog_ref);
//                }
//                else if (text::iequals(optimizer, "cgd"))
//                {
//                        res = optimize::cgd(problem, x, iterations, epsilon, fn_wlog, fn_elog, fn_ulog_ref);
//                }
//                else if (text::iequals(optimizer, "gd"))
//                {
//                        res = optimize::gd(problem, x, iterations, epsilon, fn_wlog, fn_elog, fn_ulog_ref);
//                }

//                // TODO: SGD and ASGD generic optimization methods!!!

//                else
//                {
//                        log_error() << "trainer: invalid optimization method <" << optimizer << ">!";
//                        return false;
//                }

//                model.load_params(res.optimum().x);

//                // OK
//                log_info() << "trainer: optimum [loss = " << res.optimum().f
//                           << ", gradient = " << res.optimum().g.norm()
//                           << ", calls = " << res.n_fval_calls() << "/" << res.n_grad_calls()
//                           << "], iterations = [" << res.iterations() << "/" << iterations
//                           << "].";

//                return true;
//        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
	
