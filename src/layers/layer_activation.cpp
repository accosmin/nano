#include "layer_activation.h"
#include "core/transform.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        size_t activation_layer_t::resize(size_t idims, size_t irows, size_t icols)
        {
                const size_t odims = idims;

                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, irows, icols);

                return 0;
        }

        //-------------------------------------------------------------------------------------------------

        void activation_layer_t::zero_params()
        {
        }

        //-------------------------------------------------------------------------------------------------

        void activation_layer_t::random_params(scalar_t min, scalar_t max)
        {
        }

        //-------------------------------------------------------------------------------------------------

        void activation_layer_t::zero_grad() const
        {
        }

        //-------------------------------------------------------------------------------------------------

        serializer_t& activation_layer_t::save_params(serializer_t& s) const
        {
                return s;
        }

        //-------------------------------------------------------------------------------------------------

        serializer_t& activation_layer_t::save_grad(serializer_t& s) const
        {
                return s;
        }

        //-------------------------------------------------------------------------------------------------

        deserializer_t& activation_layer_t::load_params(deserializer_t& s) const
        {
                return s;
        }

        //-------------------------------------------------------------------------------------------------

        bool activation_layer_t::save(boost::archive::binary_oarchive& oa) const
        {
                // TODO

                return false;
        }

        //-------------------------------------------------------------------------------------------------

        bool activation_layer_t::load(boost::archive::binary_iarchive& ia)
        {
                // TODO

                return false;
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& activation_layer_t::forward(const tensor3d_t& input) const
        {
                assert(n_idims() == input.n_dim1());
                assert(n_irows() <= input.n_rows());
                assert(n_icols() <= input.n_cols());

                m_idata = input;

                for (size_t o = 0; o < n_odims(); o ++)
                {
                        matrix_t& odata = m_odata(o);

                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                const matrix_t& idata = m_idata(i);

                                math::transform(idata, odata, odata,
                                        std::bind(&activation_layer_t::forward_scalar, this, _1));
                        }
                }

                return m_odata;
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& activation_layer_t::backward(const tensor3d_t& gradient) const
        {
                assert(n_odims() == gradient.n_dim1());
                assert(n_orows() == gradient.n_rows());
                assert(n_ocols() == gradient.n_cols());

                for (size_t o = 0; o < n_odims(); o ++)
                {
                        const matrix_t& odata = m_odata(o);
                        const matrix_t& gdata = gradient(o);

                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                matrix_t& idata = m_idata(i);

                                math::transform(gdata, odata, idata,
                                        std::bind(&activation_layer_t::backward_scalar, this, _1, _2));
                        }
                }

                return m_idata;
        }

        //-------------------------------------------------------------------------------------------------
}

