#ifndef NANOCV_TENSOR_STORAGE_HPP
#define NANOCV_TENSOR_STORAGE_HPP

#include <boost/serialization/access.hpp>
#include <cassert>
#include "matrix.hpp"
#include "common/random.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// store (3D/4D) tensor data as a 1D array of 2D matrices
                ///
                template
                <
                        typename tscalar,
                        typename tsize
                >
                class storage_t
                {
                public:

                        typedef typename matrix_types_t<tscalar>::tmatrix       tmatrix;
                        typedef typename matrix_types_t<tscalar>::tmatrices     tmatrices;

                        ///
                        /// \brief constructor
                        /// \param size total number of fixed size 2D matrices
                        ///
                        storage_t(tsize size = 0, tsize rows = 0, tsize cols = 0)
                        {
                                allocate(size, rows, cols);
                        }

                        ///
                        /// \brief destructor
                        ///
                        virtual ~storage_t()
                        {
                        }

                        ///
                        /// \brief fill with zero
                        ///
                        void zero()
                        {
                                for (tmatrix& mat : m_data)
                                {
                                        mat.setZero();
                                }
                        }

                        ///
                        /// \brief fill with a constant value
                        ///
                        void constant(tscalar value)
                        {
                                for (tmatrix& mat : m_data)
                                {
                                        mat.setConstant(value);
                                }
                        }

                        ///
                        /// \brief fill with uniformly distributed random values in the [min, max] range
                        ///
                        void random(tscalar min = -1, tscalar max = 1)
                        {
                                random_t<tscalar> rgen(min, max);
                                for (tmatrix& mat : m_data)
                                {
                                        rgen(mat.data(), mat.data() + mat.size());
                                }
                        }

                        ///
                        /// \brief cumulate with another tensor
                        ///
                        void operator+=(const storage_t& other)
                        {
                                assert(size() == other.size());
                                assert(n_rows() == other.n_rows());
                                assert(n_cols() == other.n_cols());

                                for (tsize i = 0; i < m_data.size(); i ++)
                                {
                                        get(i).noalias() += get(i);
                                }
                        }

                        ///
                        /// \brief access functions
                        ///
                        tsize size() const { return m_data.size() * n_rows() * n_cols(); }
                        tsize n_rows() const { return m_rows; }
                        tsize n_cols() const { return m_cols; }

                protected:

                        ///
                        /// \brief resize to new dimensions
                        ///
                        tsize allocate(tsize size, tsize rows, tsize cols)
                        {
                                m_rows = rows;
                                m_cols = cols;

                                m_data.resize(size);
                                for (tmatrix& mat : m_data)
                                {
                                        mat.resize(rows, cols);
                                        mat.setZero();
                                }

                                return this->size();
                        }

                        // access functions
                        const tmatrix& get(tsize i) const { return m_data[i]; }
                        tmatrix& get(tsize i) { return m_data[i]; }

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
                                ar & m_rows;
                                ar & m_cols;
                                ar & m_data;
                        }

                private:

                        // attributes
                        tsize           m_rows; ///< #rows (for each dimension)
                        tsize           m_cols; ///< #cols (for each dimension)
                        tmatrices       m_data; ///< values
                };
        }
}

#endif // NANOCV_TENSOR_STORAGE_HPP
