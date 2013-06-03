#include "ncv_model_affine_conv.h"
#include "ncv_logger.h"
#include "ncv_thread.h"
#include "ncv_optimize.h"
#include "ncv_timer.h"
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace ncv
{
//        //-------------------------------------------------------------------------------------------------

//        affine_conv_model_t::affine_conv_model_t(const string_t& params)
//                :       model_t("affine convolution",
//                                "parameters: iters=256[8-2048],eps=1e-5[1e-6,1e-3]")
//        {
////                opt=lbfgs[gd,cgd,lbfgs,sgd],
////                m_opt_method = text::from_params<optimization_method>(params, "opt", optimization_method::lbfgs);
//                m_opt_iters = math::clamp(text::from_params<size_t>(params, "iters", 256), 8, 2048);
//                m_opt_eps = math::clamp(text::from_params<scalar_t>(params, "eps", 1e-5), 1e-3, 1e-6);
//        }

//        //-------------------------------------------------------------------------------------------------

//        affine_conv_model_t::data_t::data_t(size_t n_outputs, size_t n_rows, size_t n_cols)
//                :       m_loss(0.0),
//                        m_count(0)
//        {
//                resize(n_outputs, n_rows, n_cols);
//        }

//        //-------------------------------------------------------------------------------------------------

//        void affine_conv_model_t::data_t::resize(size_t n_outputs, size_t n_rows, size_t n_cols)
//        {
//                m_conv.resize(n_outputs);
//                for (size_t o = 0; o < n_outputs; o ++)
//                {
//                        m_conv[o].resize(n_rows, n_cols);
//                }

//                m_bias.resize(n_outputs);
//        }

//        //-------------------------------------------------------------------------------------------------

//        vector_t affine_conv_model_t::data_t::process(const matrix_t& input) const
//        {

//        }

//        //-------------------------------------------------------------------------------------------------

//        void affine_conv_model_t::data_t::process(const matrix_t& input, const vector_t& target, const loss_t& loss)
//        {

//        }

//        //-------------------------------------------------------------------------------------------------

//        void affine_conv_model_t::data_t::operator+=(const data_t& data)
//        {

//        }

//        //-------------------------------------------------------------------------------------------------

//        size_t affine_conv_model_t::resize()
//        {

//        }

//        //-------------------------------------------------------------------------------------------------

//        vector_t affine_conv_model_t::process(const image_t& image, coord_t x, coord_t y) const
//        {
//                const matrix_t input = image.get_luma(geom::make_rect(x, y, n_cols(), n_rows()));

//                vector_t output(n_outputs());
//                for (size_t o = 0; o < n_outputs(); o ++)
//                {
//                        output(o) = m_conv[o].cwiseProduct(input).sum() + m_bias(o);
//                }

//                return output;
//        }

//        //-------------------------------------------------------------------------------------------------

//        void affine_conv_model_t::zero()
//        {
//                model_t::zero(m_conv);
//                model_t::zero(m_bias);
//        }

//        //-------------------------------------------------------------------------------------------------

//        void affine_conv_model_t::random()
//        {
//                const scalar_t min = -1.0 / sqrt(n_parameters());
//                const scalar_t max = +1.0 / sqrt(n_parameters());

//                model_t::random(min, max, m_conv);
//                model_t::random(min, max, m_bias);
//        }

//        //-------------------------------------------------------------------------------------------------

//        bool affine_conv_model_t::save(std::ofstream& os) const
//        {
//                boost::archive::binary_oarchive oa(os);
//                oa << m_conv;
//                oa << m_bias;

//                return os.good();
//        }

//        //-------------------------------------------------------------------------------------------------

//        bool affine_conv_model_t::load(std::ifstream& is)
//        {
//                boost::archive::binary_iarchive ia(is);
//                ia >> m_conv;
//                ia >> m_bias;

//                return is.good();
//        }

//        //-------------------------------------------------------------------------------------------------

//        vector_t affine_conv_model_t::to_params() const
//        {
//                vector_t params(n_parameters());

//                size_t pos = 0;
//                geom::serialize(m_conv, pos, params);
//                geom::serialize(m_bias, pos, params);

//                return params;
//        }

//        //-------------------------------------------------------------------------------------------------

//        void affine_conv_model_t::from_params(const vector_t& params)
//        {
//                size_t pos = 0;
//                geom::deserialize(m_conv, pos, params);
//                geom::deserialize(m_bias, pos, params);
//        }

//        //-------------------------------------------------------------------------------------------------

//        void affine_conv_model_t::cum_fval(const task_t& task, const loss_t& loss, const sample_t& sample,
//                opt_data_t& data) const
//        {
//                const image_t& image = task.image(sample.m_index);
//                const vector_t target = image.get_target(sample.m_region);
//                if (image.has_target(target))
//                {
//                        const vector_t input = image.get_input(sample.m_region);

//                        vector_t output;
//                        process(input, output);

//                        data.m_fx += loss.value(target, output);
//                        data.m_cnt ++;
//                }
//        }

//        //-------------------------------------------------------------------------------------------------

//        void affine_conv_model_t::cum_fval_grad(const task_t& task, const loss_t& loss, const sample_t& sample,
//                opt_data_t& data) const
//        {
//                const image_t& image = task.image(sample.m_index);
//                const vector_t target = image.get_target(sample.m_region);
//                if (image.has_target(target))
//                {
//                        const vector_t input = image.get_input(sample.m_region);

//                        vector_t output;
//                        process(input, output);

//                        vector_t lgrad;
//                        data.m_fx += loss.vgrad(target, output, lgrad);
//                        data.m_cgrad.noalias() += lgrad * input.transpose();
//                        data.m_bgrad.noalias() += lgrad;
//                        data.m_cnt ++;
//                }
//        }

//        //-------------------------------------------------------------------------------------------------

//        static void update(const optimize::result_t& result, timer_t& timer)
//        {
//                ncv::log_info() << "affine convolution model: state [loss = " << result.optimum().f
//                                << ", gradient = " << result.optimum().g.lpNorm<Eigen::Infinity>()
//                                << "] updated in " << timer.elapsed() << ".";
//                timer.start();
//        }

//        //-------------------------------------------------------------------------------------------------

//        bool affine_conv_model_t::train(const task_t& task, const samples_t& samples, const loss_t& loss)
//        {
//                // construct the optimization problem
//                auto opt_fn_size = [&] ()
//                {
//                        return n_parameters();
//                };

//                auto opt_fn_fval = [&] (const vector_t& x)
//                {
//                        from_params(x);

//                        opt_data_t cum_data(n_outputs(), n_inputs());

//                        // cumulate function value and gradients (using multiple threads)
//                        thread_loop_cumulate<opt_data_t>
//                        (
//                                samples.size(),
//                                [&] (opt_data_t& data)
//                                {
//                                        data.resize(n_outputs(), n_inputs());
//                                },
//                                [&] (size_t i, opt_data_t& data)
//                                {
//                                        cum_fval(task, loss, samples[i], data);
//                                },
//                                [&] (const opt_data_t& data)
//                                {
//                                        cum_data += data;
//                                }
//                        );

//                        math::norm(cum_data.m_fx, cum_data.m_cnt);

//                        return cum_data.m_fx;
//                };

//                auto opt_fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
//                {
//                        from_params(x);

//                        opt_data_t cum_data(n_outputs(), n_inputs());

//                        // cumulate function value and gradients (using multiple threads)
//                        thread_loop_cumulate<opt_data_t>
//                        (
//                                samples.size(),
//                                [&] (opt_data_t& data)
//                                {
//                                        data.resize(n_outputs(), n_inputs());
//                                },
//                                [&] (size_t i, opt_data_t& data)
//                                {
//                                        cum_fval_grad(task, loss, samples[i], data);
//                                },
//                                [&] (const opt_data_t& data)
//                                {
//                                        cum_data += data;
//                                }
//                        );

//                        gx.resize(n_parameters());
//                        size_t pos = 0;
//                        geom::serialize(cum_data.m_cgrad, pos, gx);
//                        geom::serialize(cum_data.m_bgrad, pos, gx);

//                        math::norm(cum_data.m_fx, cum_data.m_cnt);
//                        math::norm(gx, cum_data.m_cnt);

//                        return cum_data.m_fx;
//                };

//                const optimize::problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_fval_grad);

//                // optimize
//                timer_t timer;
//                const optimize::result_t res = optimize::lbfgs(
//                        problem, to_params(),
//                        m_opt_iters, m_opt_eps, 8,
//                        std::bind(update, _1, std::ref(timer)));

//                from_params(res.optimum().x);

//                // OK
//                log_info() << "linear model: optimum [loss = " << res.optimum().f
//                           << ", gradient = " << res.optimum().g.norm() << "]"
//                           << ", iterations = [" << res.iterations() << "/" << m_opt_iters
//                           << "], speed = [" << res.speed().avg() << " +/- " << res.speed().stdev() << "].";

//                return true;
//        }

//        //-------------------------------------------------------------------------------------------------
}

