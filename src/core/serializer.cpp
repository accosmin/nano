#include "serializer.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        serializer_t::serializer_t(vector_t& data)
                :       m_data(data),
                        m_pos(0)
        {
        }

        //-------------------------------------------------------------------------------------------------

        serializer_t& serializer_t::operator<<(const matrix_t& mat)
        {
                std::copy(mat.data(), mat.data() + mat.size(), m_data.data() + m_pos);
                m_pos += mat.size();

                return *this;
        }

        //-------------------------------------------------------------------------------------------------

        serializer_t& serializer_t::operator<<(const vector_t& vec)
        {
                std::copy(vec.data(), vec.data() + vec.size(), m_data.data() + m_pos);
                m_pos += vec.size();

                return *this;
        }

        //-------------------------------------------------------------------------------------------------

        serializer_t& serializer_t::operator<<(scalar_t val)
        {
                m_data(m_pos ++) = val;
                m_pos ++;

                return *this;
        }

        //-------------------------------------------------------------------------------------------------

        deserializer_t& deserializer_t::operator>>(matrix_t& mat)
        {
                auto segm = m_data.segment(m_pos, mat.size());
                std::copy(segm.data(), segm.data() + segm.size(), mat.data());
                m_pos += mat.size();

                return *this;
        }

        //-------------------------------------------------------------------------------------------------

        deserializer_t& deserializer_t::operator>>(vector_t& vec)
        {
                vec = m_data.segment(m_pos, vec.size());
                m_pos += vec.size();

                return *this;
        }

        //-------------------------------------------------------------------------------------------------

        deserializer_t& deserializer_t::operator>>(scalar_t& val)
        {
                val = m_data(m_pos ++);

                return *this;
        }

        //-------------------------------------------------------------------------------------------------
}

