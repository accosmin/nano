#include "serializer.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        serializer_t& serializer_t::operator<<(const matrix_t& mat)
        {
                std::copy(mat.data(), mat.data() + mat.size(), m_params.segment(m_pos, mat.size()).data());
                m_pos += mat.size();

                return *this;
        }

        //-------------------------------------------------------------------------------------------------

        serializer_t& serializer_t::operator<<(const matrices_t& mats)
        {
                for (const matrix_t& mat : mats)
                {
                        operator<<(mat);
                }

                return *this;
        }

        //-------------------------------------------------------------------------------------------------

        serializer_t& serializer_t::operator<<(const vector_t& vec)
        {
                m_params.segment(m_pos, vec.size()) = vec;
                m_pos += vec.size();

                return *this;
        }

        //-------------------------------------------------------------------------------------------------

        serializer_t& serializer_t::operator<<(scalar_t val)
        {
                m_params(m_pos ++) = val;

                return *this;
        }

        //-------------------------------------------------------------------------------------------------

        deserializer_t& deserializer_t::operator>>(matrix_t& mat)
        {
                auto segm = m_params.segment(m_pos, mat.size());
                std::copy(segm.data(), segm.data() + segm.size(), mat.data());
                m_pos += mat.size();

                return *this;
        }

        //-------------------------------------------------------------------------------------------------

        deserializer_t& deserializer_t::operator>>(matrices_t& mats)
        {
                for (matrix_t& mat : mats)
                {
                        operator>>(mat);
                }

                return *this;
        }

        //-------------------------------------------------------------------------------------------------

        deserializer_t& deserializer_t::operator>>(vector_t& vec)
        {
                vec = m_params.segment(m_pos, vec.size());
                m_pos += vec.size();

                return *this;
        }

        //-------------------------------------------------------------------------------------------------

        deserializer_t& deserializer_t::operator>>(scalar_t& val)
        {
                val = m_params(m_pos ++);

                return *this;
        }

        //-------------------------------------------------------------------------------------------------
}

