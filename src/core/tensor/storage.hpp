#ifndef NANOCV_TENSOR_STORAGE_HPP
#define NANOCV_TENSOR_STORAGE_HPP

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <cassert>
#include "core/tensor/matrix.hpp"
#include "core/random.hpp"

namespace ncv
{
        namespace tensor
        {
                /////////////////////////////////////////////////////////////////////////////////////////
                // store tensor data as a 1D array of 2D matrices.
                /////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tscalar,
                        typename tsize
                >
                class storage_t
                {
                public:

                        typedef typename matrix_types_t<tscalar>::matrix_t       matrix_t;
                        typedef typename matrix_types_t<tscalar>::matrices_t     matrices_t;

                        // constructor
                        storage_t(tsize size = 0, tsize rows = 0, tsize cols = 0)
                        {
                                allocate(size, rows, cols);
                        }

                        // destructor
                        virtual ~storage_t()
                        {
                        }

                        // reset values
                        void zero()
                        {
                                constant(0);
                        }

                        void constant(tscalar value)
                        {
                                for (matrix_t& mat : m_data)
                                {
                                        mat.setConstant(value);
                                }
                        }

                        void random(tscalar min = -1, tscalar max = 1)
                        {
                                random_t<tscalar> rgen(min, max);
                                for (matrix_t& mat : m_data)
                                {
                                        rgen(mat.data(), mat.data() + mat.size());
                                }
                        }

                        // cumulate
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

                        // access functions
                        tsize size() const { return m_data.size() * n_rows() * n_cols(); }
                        tsize n_rows() const { return m_rows; }
                        tsize n_cols() const { return m_cols; }

                protected:

                        // resize to new dimensions
                        tsize allocate(tsize size, tsize rows, tsize cols)
                        {
                                m_rows = rows;
                                m_cols = cols;

                                m_data.resize(size);
                                for (matrix_t& mat : m_data)
                                {
                                        mat.resize(rows, cols);
                                        mat.setZero();
                                }

                                return this->size();
                        }

                        // access functions
                        const matrix_t& get(tsize i) const { return m_data[i]; }
                        matrix_t& get(tsize i) { return m_data[i]; }

                private:

                        // serialize
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
                        tsize           m_rows; // #rows (for each dimension)
                        tsize           m_cols; // #cols (for each dimension)
                        matrices_t      m_data; // values
                };
        }
}

namespace boost
{
        namespace serialization
        {
                /////////////////////////////////////////////////////////////////////////////////////////
                // serialize tensor storage
                /////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        class tarchive,
                        class tscalar,
                        class tsize
                >
                void serialize(tarchive& ar, ncv::tensor::storage_t<tscalar, tsize>& st, const unsigned int version)
                {
                        ar & st;
                }
        }
}

#endif // NANOCV_TENSOR_STORAGE_HPP
