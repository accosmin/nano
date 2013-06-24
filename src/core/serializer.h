#ifndef NANOCV_SERIALIZER_H
#define NANOCV_SERIALIZER_H

#include "types.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // serialize data to vectors.
        /////////////////////////////////////////////////////////////////////////////////////////

        class serializer_t
        {
        public:

                // constructor
                serializer_t(vector_t& data);

                // serialize data
                serializer_t& operator<<(scalar_t val);
                serializer_t& operator<<(const vector_t& vec);
                serializer_t& operator<<(const matrix_t& mat);

                template
                <
                        typename tdata
                >
                serializer_t& operator<<(const std::vector<tdata>& datas)
                {
                        for (size_t i = 0; i < datas.size(); i ++)
                        {
                                (*this) << datas[i];
                        }

                        return *this;
                }

                template
                <
                        typename tdata
                >
                serializer_t& operator<<(const tdata& data)
                {
                        return (*this) << data;
                }

        private:

                // attributes
                vector_t&       m_data;
                size_t          m_pos;
        };

        class deserializer_t
        {
        public:

                // constructor
                deserializer_t(const vector_t& data)
                        :       m_data(data),
                                m_pos(0)
                {
                }

                // deserialize data
                deserializer_t& operator>>(scalar_t& val);
                deserializer_t& operator>>(vector_t& vec);
                deserializer_t& operator>>(matrix_t& mat);

                template
                <
                        typename tdata
                >
                deserializer_t& operator>>(std::vector<tdata>& datas)
                {
                        for (size_t i = 0; i < datas.size(); i ++)
                        {
                                (*this) >> datas[i];
                        }

                        return *this;
                }

                template
                <
                        typename tdata
                >
                deserializer_t& operator>>(tdata& data)
                {
                        return (*this) >> data;
                }

        private:

                // attributes
                const vector_t& m_data;
                size_t          m_pos;
        };
}

#endif // NANOCV_SERIALIZER_H
