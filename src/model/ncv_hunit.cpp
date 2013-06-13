#include "ncv_hunit.h"
#include "ncv_random.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        hunit_t::hunit_t(size_t n_inputs, size_t n_rows, size_t n_cols)
                :       m_bias(0.0),
                        m_gbias(0.0)
        {
                resize(n_inputs, n_rows, n_cols);
        }

        //-------------------------------------------------------------------------------------------------

        size_t hunit_t::resize(size_t n_inputs, size_t n_rows, size_t n_cols)
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

        void hunit_t::zero()
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

        void hunit_t::random(scalar_t min, scalar_t max)
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

        void hunit_t::serialize(serializer_t& s) const
        {
                s << m_conv << m_bias;
        }

        //-------------------------------------------------------------------------------------------------

        void hunit_t::gserialize(serializer_t& s) const
        {
                s << m_gconv << m_gbias;
        }

        //-------------------------------------------------------------------------------------------------

        void hunit_t::deserialize(deserializer_t& s)
        {
                s >> m_conv >> m_bias;
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t hunit_t::forward(const matrices_t& input) const
        {
                scalar_t result = m_bias;
                for (size_t i = 0; i < input.size(); i ++)
                {
                        result += input[i].cwiseProduct(m_conv[i]).sum();
                }

                return result;
        }

        //-------------------------------------------------------------------------------------------------

        void hunit_t::backward(const matrices_t& input, scalar_t gradient)
        {
                for (size_t i = 0; i < input.size(); i ++)
                {
                        m_gconv[i].noalias() += input[i] * gradient;
                }

                m_gbias += gradient;
        }

        //-------------------------------------------------------------------------------------------------

        void hunit_t::operator+=(const hunit_t& other)
        {
                for (size_t i = 0; i < m_gconv.size(); i ++)
                {
                        m_gconv[i] += other.m_gconv[i];
                }

                m_gbias += other.m_gbias;
        }

        //-------------------------------------------------------------------------------------------------
}

