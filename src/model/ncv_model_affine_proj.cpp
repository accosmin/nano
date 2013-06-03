#include "ncv_model_affine_proj.h"
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
//        //-------------------------------------------------------------------------------------------------

//        affine_proj_model_t::affine_proj_model_t(const string_t& params)
//                :       model_t("affine projection",
//                                "parameters: iters=256[8-2048],eps=1e-5[1e-6,1e-3]")
//        {
////                opt=lbfgs[gd,cgd,lbfgs,sgd],
////                m_opt_method = text::from_params<optimization_method>(params, "opt", optimization_method::lbfgs);
//                m_opt_iters = math::clamp(text::from_params<size_t>(params, "iters", 256), 8, 2048);
//                m_opt_eps = math::clamp(text::from_params<scalar_t>(params, "eps", 1e-5), 1e-3, 1e-6);
//        }

//        //-------------------------------------------------------------------------------------------------

//        void affine_proj_model_t::resize()
//        {
//                m_u.resize(n_outputs(), n_rows());
//                m_v.resize(n_cols(), n_outputs());
//                m_p.resize(n_outputs());
//                m_b.resize(n_outputs());
//        }

//        //-------------------------------------------------------------------------------------------------

//        vector_t affine_proj_model_t::process(const image_t& image, coord_t x, coord_t y) const
//        {
//                const matrix_t input = image.get_luma(geom::make_rect(x, y, n_cols(), n_rows()));
//                return m_u * input * m_v * m_p + m_b;
//        }

//        //-------------------------------------------------------------------------------------------------

//        void affine_proj_model_t::zero()
//        {
//                m_u.setZero();
//                m_v.setZero();
//                m_p.setZero();
//                m_b.setZero();
//        }

//        //-------------------------------------------------------------------------------------------------

//        void affine_proj_model_t::random()
//        {
//                const scalar_t min = -1.0 / sqrt(n_parameters());
//                const scalar_t max = +1.0 / sqrt(n_parameters());

//                random_t<scalar_t> rgen(min, max);

//                rgen(m_u.data(), m_u.data() + m_u.size());
//                rgen(m_v.data(), m_v.data() + m_v.size());
//                rgen(m_p.data(), m_p.data() + m_p.size());
//                rgen(m_b.data(), m_b.data() + m_b.size());
//        }

//        //-------------------------------------------------------------------------------------------------

//        bool affine_proj_model_t::save(const string_t& path) const
//        {
//                std::ofstream ofs(path, std::ios::binary);

//                boost::archive::binary_oarchive oa(ofs);
//                oa << m_u;
//                oa << m_v;
//                oa << m_p;
//                oa << m_b;

//                return ofs.good();
//        }

//        //-------------------------------------------------------------------------------------------------

//        bool affine_proj_model_t::load(const string_t& path)
//        {
//                std::ifstream ifs(path, std::ios::binary);

//                boost::archive::binary_iarchive ia(ifs);
//                ia >> m_u;
//                ia >> m_v;
//                ia >> m_p;
//                ia >> m_b;

//                return ifs.good();
//        }

//        //-------------------------------------------------------------------------------------------------

//        vector_t affine_proj_model_t::to_params() const
//        {
//                vector_t params(n_parameters());

//                size_t pos = 0;
//                geom::serialize(m_u, pos, params);
//                geom::serialize(m_v, pos, params);
//                geom::serialize(m_p, pos, params);
//                geom::serialize(m_b, pos, params);

//                return params;
//        }

//        //-------------------------------------------------------------------------------------------------

//        void affine_proj_model_t::from_params(const vector_t& params)
//        {
//                size_t pos = 0;
//                geom::deserialize(m_u, pos, params);
//                geom::deserialize(m_v, pos, params);
//                geom::deserialize(m_p, pos, params);
//                geom::deserialize(m_b, pos, params);
//        }

//        //-------------------------------------------------------------------------------------------------

//        void affine_proj_model_t::cum_fval(const task_t& task, const loss_t& loss, const sample_t& sample,
//                opt_data_t& data) const
//        {
//                const image_t& image = task.image(sample.m_index);
//                const vector_t target = image.get_target(sample.m_region);
//                if (image.has_target(target))
//                {
//                        const vector_t output = model_t::process(image, sample.m_region);

//                        data.m_fx += loss.value(target, output);
//                        data.m_cnt ++;
//                }
//        }

//        //-------------------------------------------------------------------------------------------------

//        void affine_proj_model_t::cum_fval_grad(const task_t& task, const loss_t& loss, const sample_t& sample,
//                opt_data_t& data) const
//        {
//                const image_t& image = task.image(sample.m_index);
//                const vector_t target = image.get_target(sample.m_region);
//                if (image.has_target(target))
//                {
//                        const vector_t output = model_t::process(image, sample.m_region);

//                        vector_t lgrad;
//                        data.m_fx += loss.vgrad(target, output, lgrad);
//                        data.m_ugrad.noalias() += lgrad * (input * m_v * m_p).transpose();
//                        data.m_vgrad.noalias() += (m_u * input).transpose() * lgrad * m_p.transpose();
//                        data.m_pgrad.noalias() += (m_u * input * m_v).transpose() * lgrad;
//                        data.m_bgrad.noalias() += lgrad;
//                        data.m_cnt ++;
//                }
//        }

//        //-------------------------------------------------------------------------------------------------

//        static void update(const optimize::result_t& result, timer_t& timer)
//        {
//                ncv::log_info() << "affine projection model: state [loss = " << result.optimum().f
//                                << ", gradient = " << result.optimum().g.lpNorm<Eigen::Infinity>()
//                                << "] updated in " << timer.elapsed() << ".";
//                timer.start();
//        }

//        //-------------------------------------------------------------------------------------------------

//        bool affine_proj_model_t::train(const task_t& task, const samples_t& samples, const loss_t& loss)
//        {
//                // construct the optimization problem
//                auto opt_fn_size = [&] ()
//                {
//                        return n_parameters();
//                };

//                auto opt_fn_fval = [&] (const vector_t& x)
//                {
//                        from_params(x);

//                        opt_data_t cum_data(n_outputs(), n_rows(), n_cols());

//                        // cumulate function value and gradients (using multiple threads)
//                        thread_loop_cumulate<opt_data_t>
//                        (
//                                samples.size(),
//                                [&] (opt_data_t& data)
//                                {
//                                        data.resize(n_outputs(), n_rows(), n_cols());
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

//                        opt_data_t cum_data(n_outputs(), n_rows(), n_cols());

//                        // cumulate function value and gradients (using multiple threads)
//                        thread_loop_cumulate<opt_data_t>
//                        (
//                                samples.size(),
//                                [&] (opt_data_t& data)
//                                {
//                                        data.resize(n_outputs(), n_rows(), n_cols());
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
//                        geom::serialize(cum_data.m_ugrad, pos, gx);
//                        geom::serialize(cum_data.m_vgrad, pos, gx);
//                        geom::serialize(cum_data.m_pgrad, pos, gx);
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

