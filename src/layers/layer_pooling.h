#ifndef NANOCV_POOLING_LAYER_H
#define NANOCV_POOLING_LAYER_H

#include "layer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // pooling layer:
        //      transforms a set of fixed size outputs to a single output of the same size.
        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                // weight & its derivative wrt the input: twgrad_op(x, &w, &g)
                typename twgrad_op
        >
        class pooling_layer_t : public layer_t
        {
        public:

                // destructor
                virtual ~pooling_layer_t() {}

                // resize to process new inputs, returns the number of parameters
                virtual size_t resize(size_t idims, size_t irows, size_t icols)
                {
                        return _resize(idims, irows, icols);
                }

                // reset parameters
                virtual void zero_params() {}
                virtual void random_params(scalar_t min, scalar_t max) {}
                virtual void zero_grad() const {}

                // serialize parameters & gradients
                virtual serializer_t& save_params(serializer_t& s) const { return s; }
                virtual serializer_t& save_grad(serializer_t& s) const { return s; }
                virtual deserializer_t& load_params(deserializer_t& s) { return s; }

                // process inputs (compute outputs & gradients)
                virtual const tensor3d_t& forward(const tensor3d_t& input) const { return _forward(input); }
                virtual const tensor3d_t& backward(const tensor3d_t& gradient) const { return _backward(gradient); }

                // save/load parameters to/from file
                virtual bool save(boost::archive::binary_oarchive& oa) const { return true; }
                virtual bool load(boost::archive::binary_iarchive& ia) { return true; }

                // access functions
                virtual size_t n_idims() const { return m_idata.n_dim1(); }
                virtual size_t n_irows() const { return m_idata.n_rows(); }
                virtual size_t n_icols() const { return m_idata.n_cols(); }

                virtual size_t n_odims() const { return m_odata.n_dim1(); }
                virtual size_t n_orows() const { return m_odata.n_rows(); }
                virtual size_t n_ocols() const { return m_odata.n_cols(); }

        private:

                // resize to process new inputs, returns the number of parameters
                size_t _resize(size_t idims, size_t irows, size_t icols)
                {
                        const size_t odims = 1;

                        m_idata.resize(idims, irows, icols);
                        m_odata.resize(odims, irows, icols);

                        m_udata.resize(idims, irows, icols);
                        m_vdata.resize(idims, irows, icols);
                        m_sum_udata.resize(odims, irows, icols);
                        m_sum_uidata.resize(odims, irows, icols);

                        return 0;
                }

                // output
                const tensor3d_t& _forward(const tensor3d_t& input) const
                {
                        assert(n_idims() == input.n_dim1());
                        assert(n_irows() <= input.n_rows());
                        assert(n_icols() <= input.n_cols());

                        const twgrad_op wgrad_op;
                        const size_t isize = n_irows() * n_icols();

                        matrix_t& sum_udata = m_sum_udata(0);
                        matrix_t& sum_uidata = m_sum_uidata(0);

                        sum_udata.setZero();
                        sum_uidata.setZero();

                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                const matrix_t& iidata = input(i);
                                matrix_t& idata = m_idata(i);
                                matrix_t& udata = m_udata(i);
                                matrix_t& vdata = m_vdata(i);

                                for (size_t k = 0; k < isize; k ++)
                                {
                                        scalar_t& u = udata(k);
                                        scalar_t& v = vdata(k);
                                        scalar_t& su = sum_udata(k);
                                        scalar_t& sui = sum_uidata(k);

                                        idata(k) = iidata(k);
                                        wgrad_op(idata(k), u, v);
                                        su += u;
                                        sui += u * idata(k);
                                }
                        }

                        matrix_t& odata = m_odata(0);
                        for (size_t k = 0; k < isize; k ++)
                        {
                                odata(k) = sum_uidata(k) / sum_udata(k);
                        }

                        return m_odata;
                }

                // gradient
                const tensor3d_t& _backward(const tensor3d_t& gradient) const
                {
                        assert(n_odims() == gradient.n_dim1());
                        assert(n_orows() == gradient.n_rows());
                        assert(n_ocols() == gradient.n_cols());

                        const size_t isize = n_irows() * n_icols();

                        const matrix_t& sum_udata = m_sum_udata(0);
                        const matrix_t& sum_uidata = m_sum_uidata(0);
                        const matrix_t& gdata = gradient(0);

                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                const matrix_t& udata = m_udata(i);
                                const matrix_t& vdata = m_vdata(i);
                                matrix_t& idata = m_idata(i);

                                for (size_t k = 0; k < isize; k ++)
                                {
                                        const scalar_t u = udata(k);
                                        const scalar_t v = vdata(k);
                                        const scalar_t su = sum_udata(k);
                                        const scalar_t sui = sum_uidata(k);

                                        idata(k) = gdata(k) * ((v * idata(k) + u) * su - v * sui) / (su * su);
                                }
                        }

                        return m_idata;
                }

        private:

                // attributes
                mutable tensor3d_t      m_idata;        // input buffer
                mutable tensor3d_t      m_odata;        // output buffer

                mutable tensor3d_t      m_udata;        // weights
                mutable tensor3d_t      m_vdata;        // weight derivatives
                mutable tensor3d_t      m_sum_udata;    // cumulated weights
                mutable tensor3d_t      m_sum_uidata;   // cumulated weighted inputs
        };
}

#endif // NANOCV_POOLING_LAYER_H
