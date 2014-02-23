#ifndef NANOCV_TENSOR_HPP
#define NANOCV_TENSOR_HPP

#include <boost/serialization/access.hpp>
#include "matrix.hpp"
#include <cassert>

namespace ncv
{
        namespace tensor
        {
                ///
                /// 3D tensor stored as ::dims() 2D planes of size ::rows() x ::cols()
                ///
                template
                <
                        typename tscalar_,
                        typename tsize_
                >
                class tensor_t
                {
                public:

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
                                m_data.setZero();

                                return size();
                        }

                        ///
                        /// \brief set all elements to zero
                        ///
                        void zero()
                        {
                                m_data.setZero();
                        }

                        ///
                        /// \brief set all elements to constant
                        ///
                        void constant(tscalar val)
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
                                gen(m_data.data(), m_data.data() + size());
                        }

                        ///
                        /// \brief dimensions
                        ///
                        tsize size() const { return m_data.size(); }
                        tsize dims() const { return m_dims; }
                        tsize rows() const { return m_rows; }
                        tsize cols() const { return m_cols; }                        
                        tsize plane_size() const { return rows() * cols(); }

                        ///
                        /// \brief access the tensor as a vector (size() x 1)
                        ///
                        const tvector& vector() const { return m_data; }
                        Eigen::Map<tvector> vector()
                        {
                                return Eigen::Map<tvector>(data(), size());
                        }

                        ///
                        /// \brief access the whole tensor data
                        ///
                        const tscalar* data() const { return m_data.data(); }
                        tscalar* data() { return m_data.data(); }

                        ///
                        /// \brief access the 2D planes
                        ///
                        const tscalar* plane_data(tsize i) const
                        {
                                return data() + i * plane_size();
                        }
                        tscalar* plane_data(tsize i)
                        {
                                return data() + i * plane_size();
                        }
                        Eigen::Map<tmatrix> plane_matrix(tsize i = 0)
                        {
                                return Eigen::Map<tmatrix>(plane_data(i), rows(), cols());
                        }

                        const tscalar* plane_data(tsize i1, tsize i2) const
                        {
                                return data() + plane_index(i1, i2) * plane_size();
                        }
                        tscalar* plane_data(tsize i1, tsize i2)
                        {
                                return data() + plane_index(i1, i2) * plane_size();
                        }

                        ///
                        /// \brief copy to/from another tensor (of the same size)
                        ///
                        template
                        <
                                typename ttensor
                        >
                        void copy_from(const ttensor& t)
                        {
                                assert(size() == t.size());
                                std::copy(t.data(), t.data() + t.size(), data());
                        }
                        template
                        <
                                typename ttensor
                        >
                        void copy_to(ttensor& t) const
                        {
                                assert(size() == t.size());
                                std::copy(data(), data() + size(), t.data());
                        }

                        ///
                        /// \brief copy plane to/from another tensor (of the same size)
                        ///
                        template
                        <
                                typename ttensor
                        >
                        void copy_plane_from(tsize i, const ttensor& t)
                        {
                                assert(plane_size() == t.size());
                                std::copy(t.data(), t.data() + t.size(), plane_data(i));
                        }
                        template
                        <
                                typename ttensor
                        >
                        void copy_plane_to(tsize i, ttensor& t) const
                        {
                                assert(plane_size() == t.size());
                                std::copy(plane_data(i), plane_data(i) + plane_size(), t.data());
                        }

                private:

                        ///
                        /// serialize
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

                private:

                        // attributes
                        tsize                   m_dims;         ///< #dimensions
                        tsize                   m_rows;         ///< #rows (for each dimension)
                        tsize                   m_cols;         ///< #cols (for each dimension)
                        tvector                 m_data;         ///< storage (1D vector)
                };
        }
}

#endif // NANOCV_TENSOR_HPP
