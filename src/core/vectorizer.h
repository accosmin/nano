#ifndef NANOCV_VECTORIZER_H
#define NANOCV_VECTORIZER_H

#include "types.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // serialize data to/from vectors.
        /////////////////////////////////////////////////////////////////////////////////////////

        class ovectorizer_t
        {
        public:

                // constructor
                ovectorizer_t(vector_t& data);

                // serialize data
                ovectorizer_t& operator<<(scalar_t val);
                ovectorizer_t& operator<<(const vector_t& vec);
                ovectorizer_t& operator<<(const matrix_t& mat);
                ovectorizer_t& operator<<(const tensor3d_t& t);
                ovectorizer_t& operator<<(const tensor4d_t& t);

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
        ovectorizer_t& operator<<(ovectorizer_t& s, const std::vector<tdata>& datas)
        {
                for (const tdata& data : datas)
                {
                        s << data;
                }

                return s;
        }

        class ivectorizer_t
        {
        public:

                // constructor
                ivectorizer_t(const vector_t& data);

                // deserialize data
                ivectorizer_t& operator>>(scalar_t& val);
                ivectorizer_t& operator>>(vector_t& vec);
                ivectorizer_t& operator>>(matrix_t& mat);
                ivectorizer_t& operator>>(tensor3d_t& t);
                ivectorizer_t& operator>>(tensor4d_t& t);

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
        ivectorizer_t& operator>>(ivectorizer_t& s, std::vector<tdata>& datas)
        {
                for (tdata& data : datas)
                {
                        s >> data;
                }

                return s;
        }
}

#endif // NANOCV_VECTORIZER_H
