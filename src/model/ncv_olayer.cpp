#include "ncv_olayer.h"
#include "ncv_loss.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        olayer_t::olayer_t(size_t n_outputs, size_t n_inputs, size_t n_rows, size_t n_cols)
        {
                resize(n_outputs, n_inputs, n_rows, n_cols);
        }

        //-------------------------------------------------------------------------------------------------

        size_t olayer_t::resize(size_t n_outputs, size_t n_inputs, size_t n_rows, size_t n_cols)
        {
                size_t n_params = 0;

                m_ounits.resize(n_outputs);
                for (ounit_t& ounit : m_ounits)
                {
                        n_params += ounit.resize(n_inputs, n_rows, n_cols);
                }

                m_loss = 0.0;
                m_count = 0;

                return n_params;
        }

        //-------------------------------------------------------------------------------------------------

        void olayer_t::zero()
        {
                for (ounit_t& ounit : m_ounits)
                {
                        ounit.zero();
                }

                m_loss = 0.0;
                m_count = 0;
        }

        //-------------------------------------------------------------------------------------------------

        void olayer_t::random(scalar_t min, scalar_t max)
        {
                for (ounit_t& ounit : m_ounits)
                {
                        ounit.random(min, max);
                }

                m_loss = 0.0;
                m_count = 0;
        }

        //-------------------------------------------------------------------------------------------------

        void olayer_t::serialize(serializer_t& s) const
        {
                for (const ounit_t& ounit : m_ounits)
                {
                        ounit.serialize(s);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void olayer_t::gserialize(serializer_t& s) const
        {
                for (const ounit_t& ounit : m_ounits)
                {
                        ounit.gserialize(s);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void olayer_t::deserialize(deserializer_t& s)
        {
                for (ounit_t& ounit : m_ounits)
                {
                        ounit.deserialize(s);
                }
        }

        //-------------------------------------------------------------------------------------------------

        vector_t olayer_t::forward(const matrices_t& input) const
        {
                vector_t result(m_ounits.size());
                for (size_t o = 0; o < m_ounits.size(); o ++)
                {
                        result(o) = m_ounits[o].forward(input);
                }

                return result;
        }

        //-------------------------------------------------------------------------------------------------

        void olayer_t::forward(const matrices_t& input, const vector_t& target, const loss_t& loss)
        {
                const vector_t output = forward(input);

                m_loss += loss.value(target, output);
                m_count ++;
        }

        //-------------------------------------------------------------------------------------------------

        void olayer_t::backward(const matrices_t& input, const vector_t& target, const loss_t& loss)
        {
                const vector_t output = forward(input);
                const vector_t gradient = loss.vgrad(target, output);

                for (size_t o = 0; o < m_ounits.size(); o ++)
                {
                        m_ounits[o].backward(input, gradient(o));
                }

                m_loss += loss.value(target, output);
                m_count ++;
        }

        //-------------------------------------------------------------------------------------------------

        void olayer_t::operator+=(const olayer_t& other)
        {
                for (size_t o = 0; o < m_ounits.size(); o ++)
                {
                        m_ounits[o] += other.m_ounits[o];
                }

                m_loss += other.m_loss;
                m_count += other.m_count;
        }

        //-------------------------------------------------------------------------------------------------
}

