#pragma once

#include "vector.hpp"
#include "matrix.hpp"
#include <cassert>
#include <boost/serialization/access.hpp>

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief 3D tensor stored as ::dims() 2D planes of size ::rows() x ::cols()
                ///
                template
                <
                        typename tscalar_,
                        typename tsize_
                >
                class tensor_t
                {
                public:

                        typedef tscalar_                                        Scalar; // Eigen compatible
                        typedef tscalar_                                        tscalar;
                        typedef tsize_                                          tindex;
                        typedef tsize_                                          tsize;
                        typedef typename vector_types_t<tscalar>::tvector       tvector;
                        typedef typename matrix_types_t<tscalar>::tmatrix       tmatrix;

                        ///
                        /// \brief constructor
                        ///
                        tensor_t(tsize dims = 0, tsize rows = 0, tsize cols = 0)
                        {
                                resize(dims, rows, cols);
                        }

                        ///
                        /// \brief resize to new dimensions
                        ///
                        tsize resize(tsize dims, tsize rows, tsize cols)
                        {
                                m_dims = dims;
                                m_rows = rows;
                                m_cols = cols;

                                m_data.resize(m_dims * m_rows * m_cols);

                                return size();
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
                        void random(tgenerator gen)
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
                        /// \brief access an element of the tensor in the range [0, size())
                        ///
                        tscalar operator()(tindex i) const { return m_data(i); }
                        tscalar& operator()(tindex i) { return m_data(i); }

                private:

                        ///
                        /// \brief serialize
                        ///
                        friend class boost::serialization::access;
                        template
                        <
                                class tarchive
                        >
                        void serialize(tarchive & ar, const unsigned int version)
                        {
                                ar & m_dims;
                                ar & m_rows;
                                ar & m_cols;
                                ar & m_data;
                        }

                        const tscalar* data() const
                        {
                                return m_data.data();
                        }
                        tscalar* data()
                        {
                                return m_data.data();
                        }

                        const tscalar* planeData(tsize i) const
                        {
                                return data() + i * planeSize();
                        }
                        tscalar* planeData(tsize i)
                        {
                                return data() + i * planeSize();
                        }

//                        ///
//                        /// \brief copy to/from another tensor (of the same size)
//                        ///
//                        void copy_from(const tensor_t& t)
//                        {
//                                assert(size() == t.size());
//                                copy_from(t.data());
//                        }
//                        void copy_to(tensor_t& t) const
//                        {
//                                assert(size() == t.size());
//                                copy_to(t.data());
//                        }

//                        void copy_from(const tscalar* d)
//                        {
//                                m_data = tensor::map_vector(d, size());
//                        }
//                        void copy_to(tscalar* d) const
//                        {
//                                tensor::map_vector(d, size()) = m_data;
//                        }

//                        ///
//                        /// \brief copy plane to/from another tensor (of the same size)
//                        ///
//                        void copy_plane_from(tsize i, const tensor_t& t)
//                        {
//                                assert(planeSize() == static_cast<tsize>(t.size()));
//                                assert(i < dims());
//                                copy_plane_from(i, t.data());
//                        }
//                        void copy_plane_to(tsize i, tensor_t& t) const
//                        {
//                                assert(planeSize() == t.size());
//                                assert(i < dims());
//                                copy_plane_to(i, t.data());
//                        }

//                        void copy_plane_from(tsize i, const tscalar* d)
//                        {
//                                tensor::map_vector(planeData(i), planeSize()) = tensor::map_vector(d, planeSize());
//                        }
//                        void copy_plane_to(tsize i, tscalar* d) const
//                        {
//                                tensor::map_vector(d, planeSize()) = tensor::map_vector(planeData(i), planeSize());
//                        }

                private:

                        // attributes
                        tsize                   m_dims;         ///< #dimensions
                        tsize                   m_rows;         ///< #rows (for each dimension)
                        tsize                   m_cols;         ///< #cols (for each dimension)
                        tvector                 m_data;         ///< storage (1D vector)
                };
        }
}
