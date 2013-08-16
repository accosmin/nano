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

        serializer_t& operator<<(serializer_t& s, const matrix_t& mat)
        {
                std::copy(mat.data(), mat.data() + mat.size(), s.m_data.data() + s.m_pos);
                s.m_pos += mat.size();

                return s;
        }

        //-------------------------------------------------------------------------------------------------

        serializer_t& operator<<(serializer_t& s, const vector_t& vec)
        {
                std::copy(vec.data(), vec.data() + vec.size(), s.m_data.data() + s.m_pos);
                s.m_pos += vec.size();

                return s;
        }

        //-------------------------------------------------------------------------------------------------

        serializer_t& operator<<(serializer_t& s, scalar_t val)
        {
                s.m_data(s.m_pos ++) = val;
                s.m_pos ++;

                return s;
        }

        //-------------------------------------------------------------------------------------------------

        deserializer_t::deserializer_t(const vector_t& data)
                :       m_data(data),
                        m_pos(0)
        {
        }

        //-------------------------------------------------------------------------------------------------

        deserializer_t& operator>>(deserializer_t& s, matrix_t& mat)
        {
                auto segm = s.m_data.segment(s.m_pos, mat.size());
                std::copy(segm.data(), segm.data() + segm.size(), mat.data());
                s.m_pos += mat.size();

                return s;
        }

        //-------------------------------------------------------------------------------------------------

        deserializer_t& operator>>(deserializer_t& s, vector_t& vec)
        {
                vec = s.m_data.segment(s.m_pos, vec.size());
                s.m_pos += vec.size();

                return s;
        }

        //-------------------------------------------------------------------------------------------------

        deserializer_t& operator>>(deserializer_t& s, scalar_t& val)
        {
                val = s.m_data(s.m_pos ++);

                return s;
        }

        //-------------------------------------------------------------------------------------------------
}

