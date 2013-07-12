#include "model.h"
#include "core/logger.h"
#include "core/random.h"
#include "core/optimize.h"
#include "core/timer.h"
#include <fstream>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        model_t::model_t(const string_t& description)
                :       clonable_t<model_t>(description),
                        m_rows(0),
                        m_cols(0),
                        m_outputs(0),
                        m_parameters(0),
                        m_color(color_mode::luma)
        {
        }

        //-------------------------------------------------------------------------------------------------

        bool model_t::save(const string_t& path) const
        {
                std::ofstream os(path, std::ios::binary);

                boost::archive::binary_oarchive oa(os);
                oa << m_rows;
                oa << m_cols;
                oa << m_outputs;
                oa << m_parameters;

                return save(oa);        // fixme: how to check status of the stream?!
        }

        //-------------------------------------------------------------------------------------------------

        bool model_t::load(const string_t& path)
        {
                std::ifstream is(path, std::ios::binary);

                boost::archive::binary_iarchive ia(is);
                ia >> m_rows;
                ia >> m_cols;
                ia >> m_outputs;
                ia >> m_parameters;

                return load(ia);        // fixme: how to check status of the stream?!
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::test(const task_t& task, const fold_t& fold, const loss_t& loss,
                scalar_t& lvalue, scalar_t& lerror) const
        {
                lvalue = lerror = 0.0;
                size_t cnt = 0;

                const samples_t& samples = task.samples(fold);

                for (size_t i = 0; i < samples.size(); i ++)
                {
                        const sample_t& sample = samples[i];
                        const image_t& image = task.image(sample.m_index);

                        const vector_t target = image.make_target(sample.m_region);
                        if (image.has_target(target))
                        {
                                const vector_t output = process(image, sample.m_region);

                                lvalue += loss.value(target, output);
                                lerror += loss.error(target, output);
                                ++ cnt;
                        }
                }

                lvalue /= cnt;
                lerror /= cnt;
        }

        //-------------------------------------------------------------------------------------------------

        vector_t model_t::process(const image_t& image, const rect_t& region) const
        {
                return process(image, geom::left(region), geom::top(region));
        }

        //-------------------------------------------------------------------------------------------------

        vector_t model_t::process(const image_t& image, coord_t x, coord_t y) const
        {
                return process(make_input(image, x, y));
        }

        //-------------------------------------------------------------------------------------------------

        tensor3d_t model_t::make_input(const image_t& image, coord_t x, coord_t y) const
        {
                tensor3d_t data;

                const rect_t region = geom::make_rect(x, y, n_cols(), n_rows());
                switch (m_color)
                {
                case color_mode::luma:
                        data.resize(1, n_rows(), n_cols());
                        data(0) = image.make_luma(region);
                        break;

                case color_mode::rgba:
                        data.resize(3, n_rows(), n_cols());
                        data(0) = image.make_red(region);
                        data(1) = image.make_green(region);
                        data(2) = image.make_blue(region);
                        break;
                }

                return data;
        }

        //-------------------------------------------------------------------------------------------------

        tensor3d_t model_t::make_input(const image_t& image, const rect_t& region) const
        {
                return make_input(image, geom::left(region), geom::top(region));
        }

        //-------------------------------------------------------------------------------------------------

        size_t model_t::n_inputs() const
        {
                switch (m_color)
                {
                case color_mode::rgba:
                        return 3;

                case color_mode::luma:
                default:
                        return 1;
                }
        }

        //-------------------------------------------------------------------------------------------------

        bool model_t::train(const task_t& task, const fold_t& fold, const loss_t& loss, optimizer trainer)
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "model: cannot only train models with training samples!";
                        return false;
                }

                // initialize model
                m_rows = task.n_rows();
                m_cols = task.n_cols();
                m_outputs = task.n_outputs();
                m_parameters = resize();

                random();

                log_info() << "model: parameters = " << n_parameters() << ".";

                // create training data
                data_t data(task, task.samples(fold));
                prune(data);
                if (data.m_indices.empty())
                {
                        log_error() << "model: no valid training samples!";
                        return false;
                }

                // train model
                switch (trainer)
                {
                case optimizer::lbfgs:
                case optimizer::cgd:
                        return train_batch(data, loss, trainer);

                case optimizer::sgd:
                default:
                        return train_stochastic(data, loss);
                }
        }

        //-------------------------------------------------------------------------------------------------

        static void update(const optimize::result_t& result, timer_t& timer)
        {
                ncv::log_info() << "convolution network model: state [loss = " << result.optimum().f
                                << ", gradient = " << result.optimum().g.lpNorm<Eigen::Infinity>()
                                << "] updated in " << timer.elapsed() << ".";
                timer.start();
        }

        //-------------------------------------------------------------------------------------------------

        bool model_t::train_batch(const data_t& data, const loss_t& loss, optimizer trainer)
        {
                // optimization problem: size
                auto opt_fn_size = [&] ()
                {
                        return n_parameters();
                };

                // optimization problem: function value
                auto opt_fn_fval = [&] (const vector_t& x)
                {
                        load(x);
                        return value(data, loss);
                };

                // optimization problem: function value & gradient
                auto opt_fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        load(x);
                        return vgrad(data, loss, gx);
                };

                const optimize::problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_fval_grad);

                // optimize
                vector_t x(n_parameters());
                save(x);

                timer_t timer;

                static const size_t opt_iterations = 1024;
                static const size_t opt_epsilon = 1e-5;
                static const size_t opt_history = 8;
                const auto updater = std::bind(update, _1, std::ref(timer));

                optimize::result_t res;
                switch (trainer)
                {
                case optimizer::cgd:
                        res = optimize::lbfgs(problem, x, opt_iterations, opt_epsilon, opt_history, updater);
                        break;

                case optimizer::lbfgs:
                default:
                        res = optimize::cgd(problem, x, opt_iterations, opt_epsilon, updater);
                        break;
                }

                load(res.optimum().x);

                // OK
                log_info() << "model: optimum [loss = " << res.optimum().f
                           << ", gradient = " << res.optimum().g.norm() << "]"
                           << ", iterations = [" << res.iterations() << "/" << opt_iterations
                           << "], speed = [" << res.speed().avg() << " +/- " << res.speed().stdev() << "].";

                return true;
        }

        //-------------------------------------------------------------------------------------------------

        bool model_t::train_stochastic(const data_t& data, const loss_t& loss)
        {
                // optimization problem: size
                auto opt_fn_size = [&] ()
                {
                        return n_parameters();
                };

                // optimization problem: function value
                auto opt_fn_fval = [&] (const vector_t& x)
                {
                        load(x);
                        return value(data, loss);
                };

                // optimization problem: function value & gradient
                auto opt_fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        random_t<size_t> rgen(0, data.m_indices.size() - 1);

                        data_t sdata(data.m_task, data.m_samples);
                        for (size_t i = 0; i < 1; i ++)
                        {
                                const size_t index = rgen();
                                sdata.m_indices.push_back(data.m_indices[index]);
                        }

                        load(x);
                        return vgrad(sdata, loss, gx);
                };

                const optimize::problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_fval_grad);

                // optimize
                vector_t x(n_parameters());
                save(x);

                timer_t timer;

                const size_t n_samples = data.m_indices.size();
                const scalar_t opt_epsilon = 1e-5;
                const auto updater = std::bind(update, _1, std::ref(timer));

                const optimize::result_t res = optimize::sgd(problem, x, n_samples, opt_epsilon, updater);

                load(res.optimum().x);

                // OK
                log_info() << "model: optimum [loss = " << res.optimum().f
                           << ", gradient = " << res.optimum().g.norm() << "]"
                           << ", iterations = [" << res.iterations()
                           << "], speed = [" << res.speed().avg() << " +/- " << res.speed().stdev() << "].";

                return true;
        }

        //-------------------------------------------------------------------------------------------------
}
