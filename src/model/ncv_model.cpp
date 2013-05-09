#include "ncv_model.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        void model_t::test(const task_t& task, const fold_t& fold, const loss_t& loss,
                scalar_t& lvalue, scalar_t& lerror) const
        {
                lvalue = lerror = 0.0;

                sample_t sample;
                vector_t output;
                size_t cnt = 0;

                const isamples_t& isamples = task.fold(fold);
                for (size_t s = 0; s < isamples.size(); s ++)
                {

                        task.load(isamples[s], sample);

                        if (sample.has_annotation())
                        {
                                process(sample.m_data, output);

                                lvalue += loss.value(sample.m_target, output);
                                lerror += loss.error(sample.m_target, output);
                                ++ cnt;
                        }
                }

                const scalar_t inv = (cnt == 0) ? 1.0 : 1.0 / cnt;
                lvalue *= inv;
                lerror *= inv;
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::to_params(const matrix_t& mat, size_t& pos, vector_t& params)
        {
                std::copy(mat.data(), mat.data() + mat.size(), params.segment(pos, mat.size()).data());
                pos += mat.size();
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::to_params(const vector_t& vec, size_t& pos, vector_t& params)
        {
                params.segment(pos, vec.size()) = vec;
                pos += vec.size();
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::from_params(matrix_t& mat, size_t& pos, const vector_t& params)
        {
                auto segm = params.segment(pos, mat.size());
                std::copy(segm.data(), segm.data() + segm.size(), mat.data());
                pos += mat.size();
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::from_params(vector_t& vec, size_t& pos, const vector_t& params)
        {
                vec = params.segment(pos, vec.size());
                pos += vec.size();
        }

        //-------------------------------------------------------------------------------------------------
}
