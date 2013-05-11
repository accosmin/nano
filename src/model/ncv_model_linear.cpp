#include "ncv_model_linear.h"
#include "ncv_random.h"
#include "ncv_optimize.h"
#include "ncv_logger.h"
#include "ncv_timer.h"
#include "ncv_optimize.h"
#include "ncv_thread.h"
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        linear_model_t::linear_model_t(const string_t& params)
                :       model_t("linear",
                                "linear model")
        {
        }

        //-------------------------------------------------------------------------------------------------

        void linear_model_t::resize(size_t inputs, size_t outputs)
        {
                m_weights.resize(outputs, inputs);
                m_bias.resize(outputs);
        }

        //-------------------------------------------------------------------------------------------------

        void linear_model_t::process(const vector_t& input, vector_t& output) const
        {
                output = m_weights * input + m_bias;
        }

        //-------------------------------------------------------------------------------------------------

        void linear_model_t::initZero()
        {
                m_weights.setZero();
                m_bias.setZero();
        }

        //-------------------------------------------------------------------------------------------------

        void linear_model_t::initRandom(scalar_t min, scalar_t max)
        {
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
                model_t::encode(m_weights, pos, params);
                model_t::encode(m_bias, pos, params);

                return params;
        }

        //-------------------------------------------------------------------------------------------------

        void linear_model_t::from_params(const vector_t& params)
        {
                size_t pos = 0;
                model_t::decode(m_weights, pos, params);
                model_t::decode(m_bias, pos, params);
        }

        //-------------------------------------------------------------------------------------------------

        bool linear_model_t::train(const task_t& task, const fold_t& fold, const loss_t& loss,
                size_t iters, scalar_t eps)
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "cannot only train models with training samples!";
                        return false;
                }

                resize(task.n_inputs(), task.n_outputs());

                // construct the optimization problem
                typedef std::function<size_t(void)>                                     op_size_t;
                typedef std::function<scalar_t(const vector_t&)>                        op_fval_t;
                typedef std::function<scalar_t(const vector_t&, vector_t&)>             op_fval_grad_t;
                typedef optimize::problem_t<op_size_t, op_fval_t, op_fval_grad_t>       problem_t;

                samples_t samples;
                task.load(fold, samples);

                auto opt_fn_size = [&] ()
                {
                        return n_parameters();
                };

                auto opt_fn_fval = [&] (const vector_t& x)
                {
                        // not needed!
                        return 0.0;
                };

                auto opt_fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        timer_t timer;

                        timer.start();
                        from_params(x);

                        matrix_t wgrad(n_outputs(), n_inputs());
                        vector_t bgrad(n_outputs());
                        wgrad.setZero();
                        bgrad.setZero();

                        scalar_t fx = 0.0;
                        size_t cnt = 0;

                        struct thread_data
                        {
                                scalar_t        m_fx;
                                size_t          m_cnt;
                                matrix_t        m_wgrad;
                                vector_t        m_bgrad;
                        };

                        typedef std::function<void(thread_data&)>               op_init_t;
                        typedef std::function<void(size_t, thread_data&)>       op_t;
                        typedef std::function<void(const thread_data&)>         op_cumulate_t;

                        thread_loop_cumulate<size_t, thread_data, op_init_t, op_t, op_cumulate_t>
                        (
                                samples.size(),
                                [&] (thread_data& data)
                                {
                                        data.m_fx = 0.0;
                                        data.m_cnt = 0;
                                        data.m_wgrad.resize(n_outputs(), n_inputs());
                                        data.m_bgrad.resize(n_outputs());
                                        data.m_wgrad.setZero();
                                        data.m_bgrad.setZero();
                                },
                                [&] (size_t i, thread_data& data)
                                {
                                        const sample_t& sample = samples[i];
                                        if (sample.has_annotation())
                                        {
                                                vector_t output;
                                                process(sample.m_input, output);

                                                vector_t lgrad;
                                                data.m_fx += loss.vgrad(sample.m_target, output, lgrad);
                                                data.m_wgrad.noalias() += lgrad * sample.m_input.transpose();
                                                data.m_bgrad.noalias() += lgrad;

                                                data.m_cnt ++;
                                        }
                                },
                                [&] (const thread_data& data)
                                {
                                        fx += data.m_fx;
                                        cnt += data.m_cnt;
                                        wgrad.noalias() += data.m_wgrad;
                                        bgrad.noalias() += data.m_bgrad;
                                }
                        );

                        gx.resize(n_parameters());
                        size_t pos = 0;
                        model_t::encode(wgrad, pos, gx);
                        model_t::encode(bgrad, pos, gx);

                        const scalar_t inv = (cnt == 0) ? 1.0 : 1.0 / cnt;
                        fx *= inv;
                        gx *= inv;

                        log_info() << "linear model: function value = " << fx << " (done in "
                                   << timer.elapsed_string() << ").";

                        return fx;
                };

                const problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_fval_grad, iters, eps);

                // optimize
                initRandom(-1.0, 1.0);

                optimize::lbfgs(problem, to_params());

                from_params(problem.opt_x());

                // OK
                log_info() << "linear model: optimum loss = [" << problem.opt_fx() << "]" << ".";
                log_info() << "linear model: optimum grad = [" << problem.opt_gn() << "]" << ".";
                log_info() << "linear model: evaluations = [" << problem.fevals() << " + " << problem.gevals()
                          << "], iterations = [" << problem.iterations() << "/" << problem.max_iterations()
                          << "], speed = [" << problem.speed_avg() << " +/- " << problem.speed_stdev()
                          << "].";

                // OK
                return true;
        }

        //-------------------------------------------------------------------------------------------------
}

