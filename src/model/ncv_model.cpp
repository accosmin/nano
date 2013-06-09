#include "ncv_model.h"
#include "ncv_logger.h"
#include "ncv_random.h"
#include <fstream>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        model_t::model_t(const string_t& name, const string_t& description)
                :       clonable_t<model_t>(name, description),
                        m_rows(0),
                        m_cols(0),
                        m_outputs(0),
                        m_parameters(0)
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
                                const vector_t output = forward(image, sample.m_region);

                                lvalue += loss.value(target, output);
                                lerror += loss.error(target, output);
                                ++ cnt;
                        }
                }

                lvalue /= cnt;
                lerror /= cnt;
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
                m_parameters = resize();

                random();

                return train(task, task.samples(fold), loss);
        }

        //-------------------------------------------------------------------------------------------------

        vector_t model_t::forward(const image_t& image, const rect_t& region) const
        {
                return forward(image, geom::left(region), geom::top(region));
        }

        //-------------------------------------------------------------------------------------------------
}
