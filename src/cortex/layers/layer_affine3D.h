#pragma once

#include "cortex/layer.h"

namespace cortex
{
        ///
        /// \brief fully-connected affine layer that keeps the 3D structure of the input tensor
        ///
        /// parameters:
        ///     dims    - number of output dimensions
        ///
        class affine3D_layer_t : public layer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(affine3D_layer_t, "fully-connected 3D affine layer: dims=10[1,4096]")

                // constructor
                explicit affine3D_layer_t(const string_t& parameters = string_t());

                // resize to process new tensors of the given type
                virtual tensor_size_t resize(const tensor_t& tensor) override;

                // reset parameters
                virtual void zero_params() override;
                virtual void random_params(scalar_t min, scalar_t max) override;

                // serialize parameters
                virtual scalar_t* save_params(scalar_t* params) const override;
                virtual const scalar_t* load_params(const scalar_t* params) override;

                // process inputs (compute outputs & gradients)
                virtual const tensor_t& output(const tensor_t& input) override;
                virtual const tensor_t& ginput(const tensor_t& output) override;
                virtual void gparam(const tensor_t& output, scalar_t* gradient) override;

                // access functions
                virtual tensor_size_t idims() const override { return m_idata.dims(); }
                virtual tensor_size_t irows() const override { return m_idata.rows(); }
                virtual tensor_size_t icols() const override { return m_idata.cols(); }
                virtual tensor_size_t odims() const override { return m_odata.dims(); }
                virtual tensor_size_t orows() const override { return m_odata.rows(); }
                virtual tensor_size_t ocols() const override { return m_odata.cols(); }
                virtual tensor_size_t psize() const override { return m_wdata.size() + m_bdata.size(); }

        private:

                tensor_size_t isize() const { return m_idata.size(); }
                tensor_size_t osize() const { return m_odata.size(); }

        private:

                // attributes
                tensor_t        m_idata;        ///< input buffer:      isize x irows x icols
                tensor_t        m_odata;        ///< output buffer:     osize x irows x icols
                matrix_t        m_wdata;        ///< weights:           osize x isize
                vector_t        m_bdata;        ///< bias:              osize
        };
}
