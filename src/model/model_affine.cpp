#include "model_affine.h"
#include "core/logger.h"
#include "core/thread.h"
#include "core/optimize.h"
#include "core/timer.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        affine_model_t::affine_model_t(const string_t& params)
                :       model_t("affine",
                                "parameters: proc=luma[luma,rgba]")
        {
                m_opt_proc = text::from_params<color_mode>(params, "proc", color_mode::luma);
        }

        //-------------------------------------------------------------------------------------------------

        matrices_t affine_model_t::make_input(const image_t& image, coord_t x, coord_t y) const
        {
                matrices_t data;

                const rect_t region = geom::make_rect(x, y, n_cols(), n_rows());
                switch (m_opt_proc)
                {
                case color_mode::luma:
                        data.resize(1);
                        data[0] = image.make_luma(region);
                        break;

                case color_mode::rgba:
                        data.resize(3);
                        data[0] = image.make_red(region);
                        data[1] = image.make_green(region);
                        data[2] = image.make_blue(region);
                        break;
                }

                return data;
        }

        //-------------------------------------------------------------------------------------------------

        matrices_t affine_model_t::make_input(const image_t& image, const rect_t& region) const
        {
                return make_input(image, geom::left(region), geom::top(region));
        }

        //-------------------------------------------------------------------------------------------------

        size_t affine_model_t::n_inputs() const
        {
                switch (m_opt_proc)
                {
                case color_mode::rgba:
                        return 3;

                case color_mode::luma:
                default:
                        return 1;
                }
        }

        //-------------------------------------------------------------------------------------------------

        vector_t affine_model_t::forward(const image_t& image, coord_t x, coord_t y) const
        {
                return m_olayer.forward(make_input(image, x, y));
        }

        //-------------------------------------------------------------------------------------------------

        bool affine_model_t::save(boost::archive::binary_oarchive& oa) const
        {
                oa << m_olayer;
                oa << m_opt_proc;

                return true;    // fixme: how to check status of the stream?!
        }

        //-------------------------------------------------------------------------------------------------

        bool affine_model_t::load(boost::archive::binary_iarchive& ia)
        {
                ia >> m_olayer;
                ia >> m_opt_proc;

                return true;    // fixme: how to check status of the stream?!
        }

        //-------------------------------------------------------------------------------------------------

        size_t affine_model_t::resize()
        {
                return m_olayer.resize(n_outputs(), n_inputs(), n_rows(), n_cols());
        }

        //-------------------------------------------------------------------------------------------------

        void affine_model_t::zero()
        {
                m_olayer.zero();
        }

        //-------------------------------------------------------------------------------------------------

        void affine_model_t::random()
        {
                const scalar_t min = -1.0 / sqrt(n_parameters());
                const scalar_t max = +1.0 / sqrt(n_parameters());

                m_olayer.random(min, max);
        }

        //-------------------------------------------------------------------------------------------------

        vector_t affine_model_t::serialize() const
        {
                vector_t params(n_parameters());

                serializer_t s(params);
                m_olayer.serialize(s);

                return params;
        }

        //-------------------------------------------------------------------------------------------------

        void affine_model_t::deserialize(const vector_t& params)
        {
                deserializer_t s(params);
                m_olayer.deserialize(s);
        }

        //-------------------------------------------------------------------------------------------------

        void affine_model_t::cum_loss(const task_t& task, const loss_t& loss, const sample_t& sample,
                olayer_t& data) const
        {
                const image_t& image = task.image(sample.m_index);
                const vector_t target = image.make_target(sample.m_region);
                if (image.has_target(target))
                {
                        const matrices_t input = make_input(image, sample.m_region);
                        data.forward(input, target, loss);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void affine_model_t::cum_grad(const task_t& task, const loss_t& loss, const sample_t& sample,
                olayer_t& data) const
        {
                const image_t& image = task.image(sample.m_index);
                const vector_t target = image.make_target(sample.m_region);
                if (image.has_target(target))
                {
                        const matrices_t input = make_input(image, sample.m_region);
                        data.backward(input, target, loss);
                }
        }

        //-------------------------------------------------------------------------------------------------

        static void update(const optimize::result_t& result, timer_t& timer)
        {
                ncv::log_info() << "affine convolution model: state [loss = " << result.optimum().f
                                << ", gradient = " << result.optimum().g.lpNorm<Eigen::Infinity>()
                                << "] updated in " << timer.elapsed() << ".";
                timer.start();
        }

        //-------------------------------------------------------------------------------------------------

        bool affine_model_t::train(const task_t& task, const samples_t& samples, const loss_t& loss)
        {
                // optimization problem: size
                auto opt_fn_size = [&] ()
                {
                        return n_parameters();
                };

                // optimization problem: function value
                auto opt_fn_fval = [&] (const vector_t& x)
                {
                        olayer_t cum_data(n_outputs(), n_inputs(), n_rows(), n_cols());

                        thread_loop_cumulate<olayer_t>
                        (
                                samples.size(),
                                [&] (olayer_t& data)
                                {
                                        data.resize(n_outputs(), n_inputs(), n_rows(), n_cols());
                                        deserializer_t s(x);
                                        data.deserialize(s);
                                },
                                [&] (size_t i, olayer_t& data)
                                {
                                        cum_loss(task, loss, samples[i], data);
                                },
                                [&] (const olayer_t& data)
                                {
                                        cum_data += data;
                                }
                        );

                        return cum_data.loss() / cum_data.count();
                };

                // optimization problem: function value & gradient
                auto opt_fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        olayer_t cum_data(n_outputs(), n_inputs(), n_rows(), n_cols());

                        thread_loop_cumulate<olayer_t>
                        (
                                samples.size(),
                                [&] (olayer_t& data)
                                {
                                        data.resize(n_outputs(), n_inputs(), n_rows(), n_cols());
                                        deserializer_t s(x);
                                        data.deserialize(s);
                                },
                                [&] (size_t i, olayer_t& data)
                                {
                                        cum_grad(task, loss, samples[i], data);
                                },
                                [&] (const olayer_t& data)
                                {
                                        cum_data += data;
                                }
                        );

                        gx.resize(n_parameters());
                        serializer_t s(gx);
                        cum_data.gserialize(s);
                        gx /= cum_data.count();

                        return cum_data.loss() / cum_data.count();
                };

                const optimize::problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_fval_grad);

                // optimize
                static const size_t opt_iters = 256;
                static const scalar_t opt_eps = 1e-5;
                static const size_t opt_history = 8;

                timer_t timer;
                const optimize::result_t res = optimize::lbfgs(
                        problem, serialize(),
                        opt_iters, opt_eps, opt_history,
                        std::bind(update, _1, std::ref(timer)));

                deserialize(res.optimum().x);

                // OK
                log_info() << "linear model: optimum [loss = " << res.optimum().f
                           << ", gradient = " << res.optimum().g.norm() << "]"
                           << ", iterations = [" << res.iterations() << "/" << opt_iters
                           << "], speed = [" << res.speed().avg() << " +/- " << res.speed().stdev() << "].";

                return true;
        }

        //-------------------------------------------------------------------------------------------------
}

