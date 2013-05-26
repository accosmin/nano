#include "ncv_model_linear.h"
#include "ncv_logger.h"
#include "ncv_thread.h"
#include "ncv_optimize.h"
#include "ncv_random.h"
#include "ncv_timer.h"
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        linear_model_t::linear_model_t(const string_t& params)
                :       model_t("linear",
                                "parameters: iters=256[8-2048],eps=1e-5[1e-6,1e-3]")
        {
//                opt=lbfgs[gd,cgd,lbfgs,sgd],
//                m_opt_method = text::from_params<optimization_method>(params, "opt", optimization_method::lbfgs);
                m_opt_iters = math::clamp(text::from_params<size_t>(params, "iters", 256), 8, 2048);
                m_opt_eps = math::clamp(text::from_params<scalar_t>(params, "eps", 1e-5), 1e-3, 1e-6);
        }

        //-------------------------------------------------------------------------------------------------

        void linear_model_t::resize()
        {
                m_weights.resize(n_outputs(), n_inputs());
                m_bias.resize(n_outputs());
        }

        //-------------------------------------------------------------------------------------------------

        void linear_model_t::process(const vector_t& input, vector_t& output) const
        {
                output = m_weights * input + m_bias;
        }

        //-------------------------------------------------------------------------------------------------

        void linear_model_t::zero()
        {
                m_weights.setZero();
                m_bias.setZero();
        }

        //-------------------------------------------------------------------------------------------------

        void linear_model_t::random()
        {
                const scalar_t min = -1.0 / sqrt(n_parameters());
                const scalar_t max = +1.0 / sqrt(n_parameters());

                random_t<scalar_t> rgen(min, max);

                rgen(m_weights.data(), m_weights.data() + m_weights.size());
                rgen(m_bias.data(), m_bias.data() + m_bias.size());
        }

        //-------------------------------------------------------------------------------------------------

        bool linear_model_t::save(const string_t& path) const
        {
                std::ofstream ofs(path, std::ios::binary);

                boost::archive::binary_oarchive oa(ofs);
                oa << m_weights;
                oa << m_bias;

                return ofs.good();
        }

        //-------------------------------------------------------------------------------------------------

        bool linear_model_t::load(const string_t& path)
        {
                std::ifstream ifs(path, std::ios::binary);

                boost::archive::binary_iarchive ia(ifs);
                ia >> m_weights;
                ia >> m_bias;

                return ifs.good() &&
                       static_cast<size_t>(m_bias.size()) == n_outputs();
        }

        //-------------------------------------------------------------------------------------------------

        vector_t linear_model_t::to_params() const
        {
                vector_t params(n_parameters());

                size_t pos = 0;
                geom::serialize(m_weights, pos, params);
                geom::serialize(m_bias, pos, params);

                return params;
        }

        //-------------------------------------------------------------------------------------------------

        void linear_model_t::from_params(const vector_t& params)
        {
                size_t pos = 0;
                geom::deserialize(m_weights, pos, params);
                geom::deserialize(m_bias, pos, params);
        }

        //-------------------------------------------------------------------------------------------------

        void linear_model_t::cum_fval(const task_t& task, const loss_t& loss, const sample_t& sample,
                opt_data_t& data) const
        {
                const image_t& image = task.image(sample.m_index);
                const vector_t target = image.get_target(sample.m_region);
                if (image.has_target(target))
                {
                        const vector_t input = image.get_input(sample.m_region);

                        vector_t output;
                        process(input, output);

                        data.m_fx += loss.value(target, output);
                        data.m_cnt ++;
                }
        }

        //-------------------------------------------------------------------------------------------------

        void linear_model_t::cum_fval_grad(const task_t& task, const loss_t& loss, const sample_t& sample,
                opt_data_t& data) const
        {
                const image_t& image = task.image(sample.m_index);
                const vector_t target = image.get_target(sample.m_region);
                if (image.has_target(target))
                {
                        const vector_t input = image.get_input(sample.m_region);

                        vector_t output;
                        process(input, output);

                        vector_t lgrad;
                        data.m_fx += loss.vgrad(target, output, lgrad);
                        data.m_wgrad.noalias() += lgrad * input.transpose();
                        data.m_bgrad.noalias() += lgrad;
                        data.m_cnt ++;
                }
        }

        //-------------------------------------------------------------------------------------------------

        static void update(const optimize::result_t& result, timer_t& timer)
        {
                ncv::log_info() << "linear model: state [loss = " << result.optimum().f
                                << ", gradient = " << result.optimum().g.lpNorm<Eigen::Infinity>()
                                << "] updated in " << timer.elapsed() << ".";
                timer.start();
        }

        //-------------------------------------------------------------------------------------------------

        bool linear_model_t::train(const task_t& task, const samples_t& samples, const loss_t& loss)
        {
                // construct the optimization problem
                auto opt_fn_size = [&] ()
                {
                        return n_parameters();
                };

                auto opt_fn_fval = [&] (const vector_t& x)
                {
                        from_params(x);

                        opt_data_t cum_data(n_outputs(), n_inputs());

                        // cumulate function value and gradients (using multiple threads)
                        thread_loop_cumulate<opt_data_t>
                        (
                                samples.size(),
                                [&] (opt_data_t& data)
                                {
                                        data.resize(n_outputs(), n_inputs());
                                },
                                [&] (size_t i, opt_data_t& data)
                                {
                                        cum_fval(task, loss, samples[i], data);
                                },
                                [&] (const opt_data_t& data)
                                {
                                        cum_data += data;
                                }
                        );

                        math::norm(cum_data.m_fx, cum_data.m_cnt);

                        return cum_data.m_fx;
                };

                auto opt_fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        from_params(x);

                        opt_data_t cum_data(n_outputs(), n_inputs());

                        // cumulate function value and gradients (using multiple threads)
                        thread_loop_cumulate<opt_data_t>
                        (
                                samples.size(),
                                [&] (opt_data_t& data)
                                {
                                        data.resize(n_outputs(), n_inputs());
                                },
                                [&] (size_t i, opt_data_t& data)
                                {
                                        cum_fval_grad(task, loss, samples[i], data);
                                },
                                [&] (const opt_data_t& data)
                                {
                                        cum_data += data;
                                }
                        );

                        gx.resize(n_parameters());
                        size_t pos = 0;
                        geom::serialize(cum_data.m_wgrad, pos, gx);
                        geom::serialize(cum_data.m_bgrad, pos, gx);

                        math::norm(cum_data.m_fx, cum_data.m_cnt);
                        math::norm(gx, cum_data.m_cnt);

                        return cum_data.m_fx;
                };

                const optimize::problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_fval_grad);

                // optimize
                timer_t timer;
                const optimize::result_t res = optimize::lbfgs(
                        problem, to_params(),
                        m_opt_iters, m_opt_eps, 8,
                        std::bind(update, _1, std::ref(timer)));
                from_params(res.optimum().x);

                // OK
                log_info() << "linear model: optimum [loss = " << res.optimum().f
                           << ", gradient = " << res.optimum().g.norm() << "]"
                           << ", iterations = [" << res.iterations() << "/" << m_opt_iters
                           << "], speed = [" << res.speed().avg() << " +/- " << res.speed().stdev() << "].";

                return true;
        }

        //-------------------------------------------------------------------------------------------------
}

