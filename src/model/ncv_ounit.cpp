#include "ncv_ounit.h"
#include "ncv_random.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        void serialize(const matrix_t& mat, size_t& pos, vector_t& params)
        {
                std::copy(mat.data(), mat.data() + mat.size(), params.segment(pos, mat.size()).data());
                pos += mat.size();
        }

        //-------------------------------------------------------------------------------------------------

        void serialize(const matrices_t& mats, size_t& pos, vector_t& params)
        {
                for (const matrix_t& mat : mats)
                {
                        serialize(mat, pos, params);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void serialize(const vector_t& vec, size_t& pos, vector_t& params)
        {
                params.segment(pos, vec.size()) = vec;
                pos += vec.size();
        }

        //-------------------------------------------------------------------------------------------------

        void serialize(scalar_t val, size_t& pos, vector_t& params)
        {
                params(pos ++) = val;
        }

        //-------------------------------------------------------------------------------------------------

        void deserialize(matrix_t& mat, size_t& pos, const vector_t& params)
        {
                auto segm = params.segment(pos, mat.size());
                std::copy(segm.data(), segm.data() + segm.size(), mat.data());
                pos += mat.size();
        }

        //-------------------------------------------------------------------------------------------------

        void deserialize(matrices_t& mats, size_t& pos, const vector_t& params)
        {
                for (matrix_t& mat : mats)
                {
                        deserialize(mat, pos, params);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void deserialize(vector_t& vec, size_t& pos, const vector_t& params)
        {
                vec = params.segment(pos, vec.size());
                pos += vec.size();
        }

        //-------------------------------------------------------------------------------------------------

        void deserialize(scalar_t& val, size_t& pos, const vector_t& params)
        {
                val = params(pos ++);
        }

        //-------------------------------------------------------------------------------------------------

        ounit_t::ounit_t(size_t n_inputs, size_t n_rows, size_t n_cols)
                :       m_bias(0.0),
                        m_gbias(0.0)
        {
                resize(n_inputs, n_rows, n_cols);
        }

        //-------------------------------------------------------------------------------------------------

        size_t ounit_t::resize(size_t n_inputs, size_t n_rows, size_t n_cols)
        {
                m_conv.resize(n_inputs);
                m_gconv.resize(n_inputs);

                for (matrix_t& mat : m_conv)
                {
                        mat.resize(n_rows, n_cols);
                        mat.setZero();
                }
                for (matrix_t& mat : m_gconv)
                {
                        mat.resize(n_rows, n_cols);
                        mat.setZero();
                }

                m_bias = 0.0;
                m_gbias = 0.0;

                return n_inputs * (n_rows * n_cols) + 1;
        }

        //-------------------------------------------------------------------------------------------------

        void ounit_t::zero()
        {
                for (matrix_t& mat : m_conv)
                {
                        mat.setZero();
                }
                for (matrix_t& mat : m_gconv)
                {
                        mat.setZero();
                }

                m_bias = 0.0;
                m_gbias = 0.0;
        }

        //-------------------------------------------------------------------------------------------------

        void ounit_t::random(scalar_t min, scalar_t max)
        {
                random_t<scalar_t> rgen(min, max);

                for (matrix_t& mat : m_conv)
                {
                        rgen(mat.data(), mat.data() + mat.size());
                }
                for (matrix_t& mat : m_gconv)
                {
                        mat.setZero();
                }

                m_bias = rgen();
                m_gbias = 0.0;
        }

        //-------------------------------------------------------------------------------------------------

        void ounit_t::serialize(size_t& pos, vector_t& params) const
        {
                for (const matrix_t& mat : m_conv)
                {
                        ncv::serialize(mat, pos, params);
                }

                ncv::serialize(m_bias, pos, params);
        }

        //-------------------------------------------------------------------------------------------------

        void ounit_t::deserialize(size_t& pos, const vector_t& params)
        {
                for (matrix_t& mat : m_conv)
                {
                        ncv::deserialize(mat, pos, params);
                }

                ncv::deserialize(m_bias, pos, params);
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t ounit_t::forward(const matrices_t& input) const
        {
                scalar_t result = m_bias;
                for (size_t i = 0; i < input.size(); i ++)
                {
                        result += input[i].cwiseProduct(m_conv[i]).sum();
                }

                return result;
        }

        //-------------------------------------------------------------------------------------------------

        void ounit_t::backward(const matrices_t& input, scalar_t gradient)
        {
                for (size_t i = 0; i < input.size(); i ++)
                {
                        m_gconv[i].noalias() += m_conv[i] * gradient;
                }

                m_gbias += gradient;
        }

        //-------------------------------------------------------------------------------------------------

        void ounit_t::operator+=(const ounit_t& other)
        {
                for (size_t i = 0; i < m_gconv.size(); i ++)
                {
                        m_gconv[i] += other.m_gconv[i];
                }

                m_gbias += other.m_gbias;
        }

        //-------------------------------------------------------------------------------------------------
}

