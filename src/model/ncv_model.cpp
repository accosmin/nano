#include "ncv_model.h"
#include "ncv_logger.h"
#include "ncv_random.h"
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

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

                return os.good() && save(os);
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

                return is.good() && load(is);
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::test(const task_t& task, const fold_t& fold, const loss_t& loss,
                scalar_t& lvalue, scalar_t& lerror) const
        {
                lvalue = lerror = 0.0;
                size_t cnt = 0;

                const samples_t& samples = task.samples(fold);
                foreach_sample_with_target(task, samples, [&] (size_t i, const vector_t& output, const vector_t& target)
                {
                        lvalue += loss.value(target, output);
                        lerror += loss.error(target, output);
                        ++ cnt;
                });

                math::norm(lvalue, cnt);
                math::norm(lerror, cnt);
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

        vector_t model_t::process(const image_t& image, const rect_t& region) const
        {
                return process(image, geom::left(region), geom::top(region));
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::zero(matrix_t& mat)
        {
                mat.setZero();
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::zero(matrices_t& mats)
        {
                for (matrix_t& mat : mats)
                {
                        zero(mat);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::zero(vector_t& vec)
        {
                vec.setZero();
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::random(scalar_t min, scalar_t max, matrix_t& mat)
        {
                random_t<scalar_t> rgen(min, max);
                rgen(mat.data(), mat.data() + mat.size());
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::random(scalar_t min, scalar_t max, matrices_t& mats)
        {
                for (matrix_t& mat : mats)
                {
                        random(min, max, mat);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::random(scalar_t min, scalar_t max, vector_t& vec)
        {
                random_t<scalar_t> rgen(min, max);
                rgen(vec.data(), vec.data() + vec.size());
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::serialize(const matrix_t& mat, size_t& pos, vector_t& params)
        {
                std::copy(mat.data(), mat.data() + mat.size(), params.segment(pos, mat.size()).data());
                pos += mat.size();
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::serialize(const matrices_t& mats, size_t& pos, vector_t& params)
        {
                for (const matrix_t& mat : mats)
                {
                        serialize(mat, pos, params);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::serialize(const vector_t& vec, size_t& pos, vector_t& params)
        {
                params.segment(pos, vec.size()) = vec;
                pos += vec.size();
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::deserialize(matrix_t& mat, size_t& pos, const vector_t& params)
        {
                auto segm = params.segment(pos, mat.size());
                std::copy(segm.data(), segm.data() + segm.size(), mat.data());
                pos += mat.size();
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::deserialize(matrices_t& mats, size_t& pos, const vector_t& params)
        {
                for (matrix_t& mat : mats)
                {
                        deserialize(mat, pos, params);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::deserialize(vector_t& vec, size_t& pos, const vector_t& params)
        {
                vec = params.segment(pos, vec.size());
                pos += vec.size();
        }

        //-------------------------------------------------------------------------------------------------
}
