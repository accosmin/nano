#ifndef NANOCV_TENSOR3D_H
#define NANOCV_TENSOR3D_H

#include "core/serializer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // 3D tensor:
        //      - 1D collection of fixed size matrices
        /////////////////////////////////////////////////////////////////////////////////////////

        class tensor3d_t
        {
        public:

                // constructor
                tensor3d_t(size_t dim1 = 0, size_t rows = 0, size_t cols = 0);

                // resize to new dimensions
                size_t resize(size_t dim1, size_t rows, size_t cols);

                // serialize to vector (if stored as 1x1 matrices)
                vector_t to_vector() const;
                void from_vector(const vector_t& vec);

                // serialize/deserialize data
                void serialize(serializer_t& s) const;
                void deserialize(deserializer_t& s);

                // access functions
                size_t size() const { return n_dim1() * n_rows() * n_cols(); }
                size_t n_dim1() const { return m_dim1; }
                size_t n_rows() const { return m_rows; }
                size_t n_cols() const { return m_cols; }

                const matrix_t& data(size_t d1) const { return m_data[d1]; }
                matrix_t& data(size_t d1) { return m_data[d1]; }

        private:

                friend class boost::serialization::access;
                template
                <
                        class tarchive
                >
                void serialize(tarchive & ar, const unsigned int version)
                {
                        ar & m_dim1;
                        ar & m_rows;
                        ar & m_cols;
                        ar & m_data;
                }

        private:

                // attributes
                size_t          m_dim1; // #dimension 1
                size_t          m_rows; // #rows (for each dimension)
                size_t          m_cols; // #cols (for each dimension)
                matrices_t      m_data; // values (e.g. inputs, parameters, results)
        };
}

#endif // NANOCV_TENSOR3D_H
