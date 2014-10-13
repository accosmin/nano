#pragma once

#include "layer.h"

namespace ncv
{
        ///
        /// pooling layer:
        ///     down-sample by 2 from a 3x3 neighbouring region using a soft weighting function
        ///     that approximates min/max/avg depending on the <alpha> parameter:
        ///     * min           - large negative alpha
        ///     * avg           - close to zero alpha
        ///     * max           - large positive alpha
        ///
        class pool_layer_t : public layer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(pool_layer_t, "pooling layer, parameters: alpha=[-100.0,+100.0]")

                // constructor
                pool_layer_t(const string_t& parameters = string_t());

                // resize to process new tensors of the given type
                virtual size_t resize(const tensor_t& tensor);

                // reset parameters
                virtual void zero_params() {}
                virtual void random_params(scalar_t min, scalar_t max) {}

                // serialize parameters (to memory)
                virtual scalar_t* save_params(scalar_t* params) const { return params; }
                virtual const scalar_t* load_params(const scalar_t* params) { return params; }

                // serialize parameters (to disk)
                virtual boost::archive::binary_oarchive& save(boost::archive::binary_oarchive& oa) const { return oa; }
                virtual boost::archive::binary_iarchive& load(boost::archive::binary_iarchive& ia) { return ia; }

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
                virtual size_t psize() const { return 0; }

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
                
                NANOCV_MAKE_CLONABLE(pool_max_layer_t, "pooling layer, parameters: alpha=[-100.0,+100.0]")
                
                // constructor
                pool_max_layer_t(const string_t& = string_t())
                        :       pool_layer_t("alpha=10.0")
                {
                }                
        };
        
        class pool_min_layer_t : public pool_layer_t
        {
        public:
                
                NANOCV_MAKE_CLONABLE(pool_min_layer_t, "pooling layer, parameters: alpha=[-100.0,+100.0]")
                
                // constructor
                pool_min_layer_t(const string_t& = string_t())
                        :       pool_layer_t("alpha=-10.0")
                {
                }                
        };
        
        class pool_avg_layer_t : public pool_layer_t
        {
        public:
                
                NANOCV_MAKE_CLONABLE(pool_avg_layer_t, "pooling layer, parameters: alpha=[-100.0,+100.0]")
                
                // constructor
                pool_avg_layer_t(const string_t& = string_t())
                        :       pool_layer_t("alpha=0.1")
                {
                }                
        };
}
