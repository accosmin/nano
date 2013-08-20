#ifndef NANOCV_TENSOR_HPP
#define NANOCV_TENSOR_HPP

#include "core/serializer.h"

namespace ncv
{
//        /////////////////////////////////////////////////////////////////////////////////////////
//        // 3D/4D tensor:
//        //      - 1D/2D collection of fixed size matrices
//        /////////////////////////////////////////////////////////////////////////////////////////

//        template
//        <
//                typename tmatrix
//        >
//        class tensor3d_t
//        {
//        public:

//                // constructor
//                tensor3d_t(size_t dim1 = 0, size_t rows = 0, size_t cols = 0);

//                // resize to new dimensions
//                size_t resize(size_t dim1, size_t rows, size_t cols);

//                // reset
//                void zero();
//                void constant(scalar_t value);
//                void random(scalar_t min = -0.1, scalar_t max = 0.1);

//                // serialize/deserialize data
//                friend serializer_t& operator<<(serializer_t& s, const tensor3d_t& tensor);
//                friend deserializer_t& operator>>(deserializer_t& s, tensor3d_t& tensor);

//                // cumulate
//                void operator+=(const tensor3d_t& other);

//                // access functions
//                size_t size() const { return n_dim1() * n_rows() * n_cols(); }
//                size_t n_dim1() const { return m_dim1; }
//                size_t n_rows() const { return m_rows; }
//                size_t n_cols() const { return m_cols; }

//                const matrix_t& operator()(size_t d1) const { return m_data[d1]; }
//                matrix_t& operator()(size_t d1) { return m_data[d1]; }

//        private:

//                friend class boost::serialization::access;
//                template
//                <
//                        class tarchive
//                >
//                void serialize(tarchive & ar, const unsigned int version)
//                {
//                        ar & m_dim1;
//                        ar & m_rows;
//                        ar & m_cols;
//                        ar & m_data;
//                }

//        private:

//                // attributes
//                size_t          m_dim1; // #dimension 1
//                size_t          m_rows; // #rows (for each dimension)
//                size_t          m_cols; // #cols (for each dimension)
//                matrices_t      m_data; // values
//        };

//        // serialize/deserialize data
//        serializer_t& operator<<(serializer_t& s, const tensor3d_t& tensor);
//        deserializer_t& operator>>(deserializer_t& s, tensor3d_t& tensor);
}

#endif // NANOCV_TENSOR_HPP
