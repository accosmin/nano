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
                friend serializer_t& operator<<(serializer_t& s, scalar_t val);
                friend serializer_t& operator<<(serializer_t& s, const vector_t& vec);
                friend serializer_t& operator<<(serializer_t& s, const matrix_t& mat);

        private:

                // attributes
                vector_t&       m_data;
                size_t          m_pos;
        };

        // serialize data
        serializer_t& operator<<(serializer_t& s, scalar_t val);
        serializer_t& operator<<(serializer_t& s, const vector_t& vec);
        serializer_t& operator<<(serializer_t& s, const matrix_t& mat);

        // serialize vectors of known types
        template
        <
                typename tdata
        >
        serializer_t& operator<<(serializer_t& s, const std::vector<tdata>& datas)
        {
                for (const tdata& data : datas)
                {
                        s << data;
                }

                return s;
        }

        class deserializer_t
        {
        public:

                // constructor
                deserializer_t(const vector_t& data);

                // deserialize data
                friend deserializer_t& operator>>(deserializer_t& s, scalar_t& val);
                friend deserializer_t& operator>>(deserializer_t& s, vector_t& vec);
                friend deserializer_t& operator>>(deserializer_t& s, matrix_t& mat);

        private:

                // attributes
                const vector_t& m_data;
                size_t          m_pos;
        };

        // deserialize data
        deserializer_t& operator>>(deserializer_t& s, scalar_t& val);
        deserializer_t& operator>>(deserializer_t& s, vector_t& vec);
        deserializer_t& operator>>(deserializer_t& s, matrix_t& mat);

        // deserialize vectors of known types
        template
        <
                typename tdata
        >
        deserializer_t& operator>>(deserializer_t& s, std::vector<tdata>& datas)
        {
                for (tdata& data : datas)
                {
                        s >> data;
                }

                return s;
        }
}

#endif // NANOCV_SERIALIZER_H
