#ifndef NANOCV_TENSOR3D_H
#define NANOCV_TENSOR3D_H

#include "core/serializer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // 3D tensor:
        //      - 1D collection of fixed size (convolution) matrices
        //      - output is the sum of the Hadamard product of the convolutions with the inputs
        /////////////////////////////////////////////////////////////////////////////////////////

        class tensor3d_t
        {
        public:

                // constructor
                tensor3d_t(size_t dim1 = 0, size_t rows = 0, size_t cols = 0);

                // resize to new dimensions
                size_t resize(size_t dim1, size_t rows, size_t cols);

                // reset parameters
                void zero();
                void random(scalar_t min = -0.1, scalar_t max = 0.1);

                // process inputs (compute outputs & gradients)
                scalar_t forward(const tensor3d_t& input) const;
                tensor3d_t backward(scalar_t gradient) const;
                tensor3d_t gradient(const tensor3d_t& input, scalar_t gradient) const;

                // serialize/deserialize parameters
                void serialize(serializer_t& s) const;
                void deserialize(deserializer_t& s);

                // cumulate
                void operator+=(const tensor3d_t& other);

                // access functions
                size_t size() const { return dim1() * rows() * cols(); }
                size_t dim1() const { return m_dim1; }
                size_t rows() const { return m_rows; }
                size_t cols() const { return m_cols; }

                const matrix_t& operator[](size_t d1) const { return m_data[d1]; }

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
