#pragma once

#include "vector.hpp"
#include "matrix.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief 3D tensor stored as ::dims() 2D planes of size ::rows() x ::cols()
                ///
                template
                <
                        typename tvector
                >
                class tensor_base_t
                {
                public:

                        typedef typename tvector::Scalar        tscalar;
                        typedef typename tvector::Index         tindex;
                        typedef tindex                          tsize;

                        // Eigen compatible
                        typedef tscalar         Scalar;
                        typedef tindex          Index;

                        ///
                        /// \brief constructor
                        ///
                        tensor_base_t(tsize dims, tsize rows, tsize cols)
                                :       m_dims(dims),
                                        m_rows(rows),
                                        m_cols(cols)
                        {
                        }

                        ///
                        /// \brief constructor
                        ///
                        tensor_base_t(tsize dims, tsize rows, tsize cols, const tvector& data)
                                :       m_dims(dims),
                                        m_rows(rows),
                                        m_cols(cols),
                                        m_data(data)
                        {
                                assert(m_data.size() == dims * rows * cols);
                        }

                        ///
                        /// \brief destructor
                        ///
                        virtual ~tensor_base_t()
                        {
                        }

                        ///
                        /// \brief set all elements to zero
                        ///
                        void setZero()
                        {
                                m_data.setZero();
                        }

                        ///
                        /// \brief set all elements to constant
                        ///
                        void setConstant(tscalar val)
                        {
                                m_data.setConstant(val);
                        }

                        ///
                        /// \brief set all elements to random values using the given generator
                        ///
                        template
                        <
                                typename tgenerator
                        >
                        void setRandom(tgenerator gen)
                        {
                                gen(data(), data() + size());
                        }

                        ///
                        /// \brief dimensions
                        ///
                        tsize size() const { return m_data.size(); }
                        tsize dims() const { return m_dims; }
                        tsize rows() const { return m_rows; }
                        tsize cols() const { return m_cols; }
                        tsize planeSize() const { return rows() * cols(); }

                        ///
                        /// \brief access the whole tensor as a vector (size() x 1)
                        ///
                        const tvector& vector() const { return m_data; }
                        decltype(auto) vector() { return tensor::map_vector(data(), size()); }

                        ///
                        /// \brief access the whole tensor as an array
                        ///
                        const tscalar* data() const { return m_data.data(); }
                        tscalar* data() { return m_data.data(); }

                        ///
                        /// \brief access the 2D plane (i) as vector
                        ///
                        decltype(auto) vector(tindex i) const { return tensor::map_vector(planeData(i), planeSize()); }
                        decltype(auto) vector(tindex i) { return tensor::map_vector(planeData(i), planeSize()); }

                        ///
                        /// \brief access the 2D plane (i) as matrix
                        ///
                        decltype(auto) matrix(tindex i) const { return tensor::map_matrix(planeData(i), rows(), cols()); }
                        decltype(auto) matrix(tindex i) { return tensor::map_matrix(planeData(i), rows(), cols()); }

                        ///
                        /// \brief access the 2D plane (i) as an array
                        ///
                        const tscalar* planeData(tindex i) const { return data() + i * planeSize(); }
                        tscalar* planeData(tindex i) { return data() + i * planeSize(); }

                        ///
                        /// \brief access an element of the tensor in the range [0, size())
                        ///
                        tscalar operator()(tindex i) const { return m_data(i); }
                        tscalar& operator()(tindex i) { return m_data(i); }

                protected:

                        // attributes
                        tsize           m_dims;         ///< #dimensions
                        tsize           m_rows;         ///< #rows (for each dimension)
                        tsize           m_cols;         ///< #cols (for each dimension)
                        tvector         m_data;         ///< storage (1D vector)
                };
        }
}
