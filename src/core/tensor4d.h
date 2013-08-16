#ifndef NANOCV_TENSOR4D_H
#define NANOCV_TENSOR4D_H

#include "core/serializer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // 4D tensor:
        //      - 2D collection of fixed size (convolution) matrices
        /////////////////////////////////////////////////////////////////////////////////////////

        class tensor4d_t;
        typedef std::vector<tensor4d_t>         tensor4ds_t;

        class tensor4d_t
        {
        public:

                // constructor
                tensor4d_t(size_t dim1 = 0, size_t dim2 = 0, size_t rows = 0, size_t cols = 0);

                // resize to new dimensions
                size_t resize(size_t dim1, size_t dim2, size_t rows, size_t cols);

                // reset
                void zero();
                void constant(scalar_t value);
                void random(scalar_t min = -0.1, scalar_t max = 0.1);

                // serialize/deserialize data
                friend serializer_t& operator<<(serializer_t& s, const tensor4d_t& tensor);
                friend deserializer_t& operator>>(deserializer_t& s, tensor4d_t& tensor);

                // cumulate
                void operator+=(const tensor4d_t& other);

                // access functions
                size_t size() const { return n_dim1() * n_dim2() * n_rows() * n_cols(); }
                size_t n_dim1() const { return m_dim1; }
                size_t n_dim2() const { return m_dim2; }
                size_t n_rows() const { return m_rows; }
                size_t n_cols() const { return m_cols; }

                const matrix_t& operator()(size_t d1, size_t d2) const { return m_data[d1 * n_dim2() + d2]; }
                matrix_t& operator()(size_t d1, size_t d2) { return m_data[d1 * n_dim2() + d2]; }

        private:

                friend class boost::serialization::access;
                template
                <
                        class tarchive
                >
                void serialize(tarchive & ar, const unsigned int version)
                {
                        ar & m_dim1;
                        ar & m_dim2;
                        ar & m_rows;
                        ar & m_cols;
                        ar & m_data;
                }

        private:

                // attributes
                size_t          m_dim1; // #dimension 1
                size_t          m_dim2; // #dimension 2
                size_t          m_rows; // #rows (for each dimension)
                size_t          m_cols; // #cols (for each dimension)
                matrices_t      m_data; // values
        };

        // serialize/deserialize data
        serializer_t& operator<<(serializer_t& s, const tensor4d_t& tensor);
        deserializer_t& operator>>(deserializer_t& s, tensor4d_t& tensor);
}

#endif // NANOCV_TENSOR4D_H
