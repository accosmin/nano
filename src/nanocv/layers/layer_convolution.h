#pragma once

#include "layer.h"

namespace ncv
{
        ///
        /// \brief convolution layer
        ///
        /// parameters:
        ///     dims=16[1,256]          - number of convolutions (output dimension)
        ///     rows=8[1,32]            - convolution size
        ///     cols=8[1,32]            - convolution size
        ///     mask=1[0,100]           - random masking percentage (number of input planes connected per output plane)
        ///
        class conv_layer_t : public layer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(conv_layer_t,
                                     "convolution layer, "\
                                     "parameters: dims=16[1,256],rows=8[1,32],cols=8[1,32],mask=100[1,100]")

                // constructor
                conv_layer_t(const string_t& parameters = string_t());

                // destructor
                virtual ~conv_layer_t();

                // resize to process new tensors of the given type
                virtual size_t resize(const tensor_t& tensor);

                // reset parameters
                virtual void zero_params();
                virtual void random_params(scalar_t min, scalar_t max);

                // serialize parameters
                virtual scalar_t* save_params(scalar_t* params) const;
                virtual const scalar_t* load_params(const scalar_t* params);

                // serialize parameters (to disk)
                virtual boost::archive::binary_oarchive& save(boost::archive::binary_oarchive& oa) const;
                virtual boost::archive::binary_iarchive& load(boost::archive::binary_iarchive& ia);

                // process inputs (compute outputs & gradients)
                virtual const tensor_t& output(const tensor_t& input);
                virtual const tensor_t& igrad(const tensor_t& output);
                virtual void pgrad(const tensor_t& output, scalar_t* gradient);

                // access functions
                virtual size_t idims() const { return m_idata.dims(); }
                virtual size_t irows() const { return m_idata.rows(); }
                virtual size_t icols() const { return m_idata.cols(); }
                virtual size_t odims() const { return m_odata.dims(); }
                virtual size_t orows() const { return m_odata.rows(); }
                virtual size_t ocols() const { return m_odata.cols(); }
                virtual size_t psize() const;

        private:

                size_t krows() const { return m_kdata.rows(); }
                size_t kcols() const { return m_kdata.cols(); }

                void make_mask();

        private:

                // attributes
                tensor_t                m_idata;        ///< input buffer:              idims x irows x icols
                tensor_t                m_odata;        ///< output buffer:             odims x orows x ocols
                tensor_t                m_kdata;        ///< convolution kernels:       odims x idims x krows x kcols

                matrix_t                m_mdata;        ///< {0, 1} mask buffer:        odims x idims
        };
}
