#pragma once

#include "libnanocv/layer.h"

namespace ncv
{
        ///
        /// \brief pooling layer:
        ///     down-sample by 2 from a 3x3 neighbouring region using a soft weighting function
        ///     that approximates min/max/avg depending on the <alpha> parameter:
        ///     * min           - large negative alpha
        ///     * avg           - close to zero alpha
        ///     * max           - large positive alpha
        ///
        /// the soft-min/max/avg approximation is performed like described in:
        ///     http://www.johndcook.com/blog/2010/01/13/soft-maximum/
        ///
        class pool_layer_t : public layer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(pool_layer_t, "pooling layer, parameters: alpha=[-100.0,+100.0]")

                // constructor
                explicit pool_layer_t(const string_t& parameters = string_t());

                // resize to process new tensors of the given type
                virtual size_t resize(const tensor_t& tensor) override;

                // reset parameters
                virtual void zero_params() override {}
                virtual void random_params(scalar_t min, scalar_t max) override {}

                // serialize parameters (to memory)
                virtual scalar_t* save_params(scalar_t* params) const override { return params; }
                virtual const scalar_t* load_params(const scalar_t* params) override { return params; }

                // serialize parameters (to disk)
                virtual boost::archive::binary_oarchive& save(boost::archive::binary_oarchive& oa) const override { return oa; }
                virtual boost::archive::binary_iarchive& load(boost::archive::binary_iarchive& ia) override { return ia; }

                // process inputs (compute outputs & gradients)
                virtual const tensor_t& output(const tensor_t& input) override;
                virtual const tensor_t& ginput(const tensor_t& output) override;
                virtual void gparam(const tensor_t& output, scalar_t* gradient) override;

                // access functions
                virtual size_t idims() const override { return m_idata.dims(); }
                virtual size_t irows() const override { return m_idata.rows(); }
                virtual size_t icols() const override { return m_idata.cols(); }
                virtual size_t odims() const override { return m_odata.dims(); }
                virtual size_t orows() const override { return m_odata.rows(); }
                virtual size_t ocols() const override { return m_odata.cols(); }
                virtual size_t psize() const override { return 0; }

                // flops
                virtual size_t output_flops() const override { return idims() * irows() * icols() * 16; }
                virtual size_t ginput_flops() const override { return idims() * irows() * icols(); }
                virtual size_t gparam_flops() const override { return 0; }

        private:

                // attributes
                scalar_t                m_alpha;        ///< scaling factor (controls the min/max/avg-like effect)
                tensor_t                m_idata;        ///< input buffer
                tensor_t                m_odata;        ///< output buffer

                tensor_t                m_wdata;       	///< weights buffer: exp(input)
                tensor_t                m_sdata;    	///< sum buffer: cumulated exponents / output pixel    		
		tensor_t		m_cdata;	///< counts buffer: #hits / output pixel
        };
        
        class pool_max_layer_t : public pool_layer_t
        {
        public:
                
                NANOCV_MAKE_CLONABLE(pool_max_layer_t, "soft-max pooling layer")
                
                // constructor
                pool_max_layer_t(const string_t& = string_t())
                        :       pool_layer_t("alpha=10.0")
                {
                }                
        };
        
        class pool_min_layer_t : public pool_layer_t
        {
        public:
                
                NANOCV_MAKE_CLONABLE(pool_min_layer_t, "soft-min pooling layer")
                
                // constructor
                pool_min_layer_t(const string_t& = string_t())
                        :       pool_layer_t("alpha=-10.0")
                {
                }                
        };
        
        class pool_avg_layer_t : public pool_layer_t
        {
        public:
                
                NANOCV_MAKE_CLONABLE(pool_avg_layer_t, "average pooling layer")
                
                // constructor
                pool_avg_layer_t(const string_t& = string_t())
                        :       pool_layer_t("alpha=0.1")
                {
                }                
        };
}
