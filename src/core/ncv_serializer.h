#ifndef NANOCV_SERIALIZER_H
#define NANOCV_SERIALIZER_H

#include "ncv_types.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // utilities for building processing units.
        /////////////////////////////////////////////////////////////////////////////////////////

        class serializer_t
        {
        public:

                // constructor
                serializer_t(vector_t& params)
                        :       m_params(params),
                                m_pos(0)
                {
                }

                // serialize data
                serializer_t& operator<<(scalar_t val);
                serializer_t& operator<<(const vector_t& vec);
                serializer_t& operator<<(const matrix_t& mat);
                serializer_t& operator<<(const matrices_t& mats);

        private:

                // attributes
                vector_t&       m_params;
                size_t          m_pos;
        };

        class deserializer_t
        {
        public:

                // constructor
                deserializer_t(const vector_t& params)
                        :       m_params(params),
                                m_pos(0)
                {
                }

                // deserialize data
                deserializer_t& operator>>(scalar_t& val);
                deserializer_t& operator>>(vector_t& vec);
                deserializer_t& operator>>(matrix_t& mat);
                deserializer_t& operator>>(matrices_t& mats);

        private:

                // attributes
                const vector_t& m_params;
                size_t          m_pos;
        };
}

#endif // NANOCV_SERIALIZER_H
