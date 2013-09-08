#ifndef NANOCV_TENSOR_HPP
#define NANOCV_TENSOR_HPP

#include <vector>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <cassert>
#include "random.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // 3D/4D tensor:
        //      - 1D/2D collection of fixed size matrices
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace impl
        {
                template
                <
                        typename tmatrix,
                        typename tsize
                >
                class tensor_storage_t
                {
                public:

                        typedef std::vector<tmatrix>            matrices_t;
                        typedef typename tmatrix::Scalar        tscalar;

                        // constructor
                        tensor_storage_t(tsize size = 0, tsize rows = 0, tsize cols = 0)
                        {
                                allocate(size, rows, cols);
                        }

                        // destructor
                        virtual ~tensor_storage_t()
                        {
                        }

                        // reset values
                        void zero()
                        {
                                constant(0);
                        }

                        template
                        <
                                typename tscalar
                        >
                        void constant(tscalar value)
                        {
                                for (tmatrix& mat : m_data)
                                {
                                        mat.setConstant(value);
                                }
                        }

                        template
                        <
                                typename tscalar
                        >
                        void random(tscalar min = -0.1, tscalar max = 0.1)
                        {
                                random_t<tscalar> rgen(min, max);
                                for (tmatrix& mat : m_data)
                                {
                                        rgen(mat.data(), mat.data() + mat.size());
                                }
                        }

                        // cumulate
                        void operator+=(const tensor_storage_t& other)
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

        template
        <
                typename tmatrix,
                typename tsize
        >
        class ttensor3d_t : public impl::tensor_storage_t<tmatrix, tsize>
        {
        public:

                typedef impl::tensor_storage_t<tmatrix, tsize>  tbase;
                typedef typename tbase::tscalar                tscalar;

                // constructor
                ttensor3d_t(tsize dim1 = 0, tsize rows = 0, tsize cols = 0)
                {
                        resize(dim1, rows, cols);
                }

                // resize to new dimensions
                tsize resize(tsize dim1, tsize rows, tsize cols)
                {
                        m_dim1 = dim1;
                        return tbase::allocate(dim1, rows, cols);
                }

                // access functions
                tsize n_dim1() const { return m_dim1; }

                const tmatrix& operator()(tsize d1) const
                {
                        return tbase::get(d1);
                }
                tmatrix& operator()(tsize d1)
                {
                        return tbase::get(d1);
                }

                const tscalar& operator()(tsize d1, size_t r, size_t c) const
                {
                        return tbase::get(d1)(r, c);
                }
                tscalar& operator()(tsize d1, size_t r, size_t c)
                {
                        return tbase::get(d1)(r, c);
                }

        private:

                // serialize
                friend class boost::serialization::access;
                template
                <
                        class tarchive
                >
                void serialize(tarchive & ar, const unsigned int version)
                {
                        ar & boost::serialization::base_object<tbase>(*this);
                        ar & m_dim1;
                }

        private:

                // attributes
                tsize           m_dim1; // #dimension 1
        };

        template
        <
                typename tmatrix,
                typename tsize
        >
        class ttensor4d_t : public impl::tensor_storage_t<tmatrix, tsize>
        {
        public:

                typedef impl::tensor_storage_t<tmatrix, tsize>  tbase;
                typedef typename tbase::tscalar                tscalar;

                // constructor
                ttensor4d_t(tsize dim1 = 0, tsize dim2 = 0, tsize rows = 0, tsize cols = 0)
                {
                        resize(dim1, dim2, rows, cols);
                }

                // resize to new dimensions
                tsize resize(tsize dim1, tsize dim2, tsize rows, tsize cols)
                {
                        m_dim1 = dim1;
                        m_dim2 = dim2;
                        return tbase::allocate(dim1 * dim2, rows, cols);
                }

                // access functions
                tsize n_dim1() const { return m_dim1; }
                tsize n_dim2() const { return m_dim2; }

                const tmatrix& operator()(size_t d1, size_t d2) const
                {
                        return tbase::get(d1 * n_dim2() + d2);
                }
                tmatrix& operator()(size_t d1, size_t d2)
                {
                        return tbase::get(d1 * n_dim2() + d2);
                }

                const tscalar& operator()(tsize d1, tsize d2, size_t r, size_t c) const
                {
                        return tbase::get(d1, d2)(r, c);
                }
                tscalar& operator()(tsize d1, tsize d2, size_t r, size_t c)
                {
                        return tbase::get(d1, d2)(r, c);
                }

        private:

                // serialize
                friend class boost::serialization::access;
                template
                <
                        class tarchive
                >
                void serialize(tarchive & ar, const unsigned int version)
                {
                        ar & boost::serialization::base_object<tbase>(*this);
                        ar & m_dim1;
                        ar & m_dim2;
                }

        private:

                // attributes
                tsize           m_dim1; // #dimension 1
                tsize           m_dim2; // #dimension 2
        };
}

#endif // NANOCV_TENSOR_HPP
