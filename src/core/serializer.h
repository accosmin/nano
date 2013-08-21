#ifndef NANOCV_SERIALIZER_H
#define NANOCV_SERIALIZER_H

#include "types.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // serialize data to vectors.
        /////////////////////////////////////////////////////////////////////////////////////////

        // FIXME: better name for these classes! vectorize?! linearize?!

        class serializer_t
        {
        public:

                // constructor
                serializer_t(vector_t& data);

                // serialize data
                serializer_t& operator<<(scalar_t val);
                serializer_t& operator<<(const vector_t& vec);
                serializer_t& operator<<(const matrix_t& mat);
                serializer_t& operator<<(const tensor3d_t& t);
                serializer_t& operator<<(const tensor4d_t& t);

        private:

                // attributes
                vector_t&       m_data;
                size_t          m_pos;
        };

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
                deserializer_t& operator>>(scalar_t& val);
                deserializer_t& operator>>(vector_t& vec);
                deserializer_t& operator>>(matrix_t& mat);
                deserializer_t& operator>>(tensor3d_t& t);
                deserializer_t& operator>>(tensor4d_t& t);

        private:

                // attributes
                const vector_t& m_data;
                size_t          m_pos;
        };

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
