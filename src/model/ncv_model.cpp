#include "ncv_model.h"
#include "ncv_logger.h"
#include "ncv_random.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        void model_t::test(const task_t& task, const fold_t& fold, const loss_t& loss,
                scalar_t& lvalue, scalar_t& lerror) const
        {
                lvalue = lerror = 0.0;
                size_t cnt = 0;

                const samples_t& samples = task.samples(fold);
                foreach_sample_with_target(task, samples, [&] (size_t i, const vector_t& input, const vector_t& target)
                {
                        vector_t output;
                        process(input, output);

                        lvalue += loss.value(target, output);
                        lerror += loss.error(target, output);
                        ++ cnt;
                });

                math::norm(lvalue, cnt);
                math::norm(lerror, cnt);
        }

        //-------------------------------------------------------------------------------------------------

        samples_t model_t::bootstrap(const task_t& task, const samples_t& samples, const loss_t& loss,
                scalar_t factor, scalar_t& error) const
        {
                scalars_t errors;
                indices_t indices;
                size_t cnt = 0;

                foreach_sample_with_target(task, samples, [&] (size_t i, const vector_t& input, const vector_t& target)
                {
                        vector_t output;
                        process(input, output);

                        errors.push_back(loss.error(target, output));
                        indices.push_back(i);
                        cnt ++;
                });

                error = std::accumulate(errors.begin(), errors.end(), 0.0);
                math::norm(error, cnt);

                random_t<scalar_t> rgen(0.0, 1.0);
                const scalar_t prob_scale = factor * cnt / (error * cnt + std::numeric_limits<scalar_t>::epsilon());

                samples_t esamples;
                for (size_t ii = 0; ii < indices.size(); ii ++)
                {
                        const scalar_t prob = errors[ii] * prob_scale;
                        if (rgen() < prob)
                        {
                                const size_t i = indices[ii];
                                esamples.push_back(samples[i]);
                        }
                }

                return esamples;
        }

        //-------------------------------------------------------------------------------------------------

        bool model_t::train(const task_t& task, const fold_t& fold, const loss_t& loss)
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "cannot only train models with training samples!";
                        return false;
                }

                m_rows = task.n_rows();
                m_cols = task.n_cols();
                m_outputs = task.n_outputs();

                resize();
                random();

                const scalar_t boot_factor = 0.01;
                const size_t boot_steps = 8;

                samples_t samples;
                for (size_t b = 0; b < boot_steps; b ++)
                {
                        scalar_t error;
                        const samples_t esamples = bootstrap(task, task.samples(fold), loss, boot_factor, error);
                        samples.insert(samples.end(), esamples.begin(), esamples.end());

                        log_info() << "boostrap [" << (b + 1) << "/" << boot_steps
                                   << "]: error = " << error << ", samples = " << samples.size() << ".";
                        if (!train(task, samples, loss))
                        {
                                log_error() << "failed to train the model!";
                                return false;
                        }
                }

                return true;
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::process(const image_t& image, coord_t x, coord_t y, vector_t& output) const
        {
                const vector_t input = image.get_input(geom::make_rect(x, y, n_cols(), n_rows()));
                return process(input, output);
        }

        //-------------------------------------------------------------------------------------------------
}
