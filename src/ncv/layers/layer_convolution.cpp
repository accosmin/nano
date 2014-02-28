#include "layer_convolution.h"
#include "text.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "common/sum.hpp"
#include "common/convolution.hpp"
#include "common/random.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        conv_layer_t::conv_layer_t(const string_t& params)
                :       m_params(params)
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t conv_layer_t::resize(const tensor_t& tensor)
        {
                const size_t idims = tensor.dims();
                const size_t irows = tensor.rows();
                const size_t icols = tensor.cols();

                const size_t odims = math::clamp(text::from_params<size_t>(m_params, "dims", 16), 1, 256);
                const size_t crows = math::clamp(text::from_params<size_t>(m_params, "rows", 8), 1, 32);
                const size_t ccols = math::clamp(text::from_params<size_t>(m_params, "cols", 8), 1, 32);

                if (irows < crows || icols < ccols)
                {
                        const string_t message =
                                "invalid size (" + text::to_string(idims) + "x" + text::to_string(irows) +
                                 "x" + text::to_string(icols) + ") -> (" + text::to_string(odims) + "x" +
                                 text::to_string(crows) + "x" + text::to_string(ccols) + ")";

                        log_error() << "convolution layer: " << message;
                        throw std::runtime_error("convolution layer: " + message);
                }

                const size_t orows = irows - crows + 1;
                const size_t ocols = icols - ccols + 1;

                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);

                m_kdata.resize(odims, crows, ccols);
                m_wdata.resize(1, odims, idims);

                m_gkdata.resize(odims, crows, ccols);
                m_gwdata.resize(1, odims, idims);
                m_gidata.resize(idims, irows, icols);

                return m_kdata.size() + m_wdata.size();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void conv_layer_t::zero_params()
        {
                m_kdata.zero();
                m_wdata.zero();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void conv_layer_t::random_params(scalar_t min, scalar_t max)
        {
                m_kdata.random(random_t<scalar_t>(min, max));
                m_wdata.random(random_t<scalar_t>(min, max));
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ovectorizer_t& conv_layer_t::save_params(ovectorizer_t& s) const
        {
                return s << m_kdata << m_wdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ovectorizer_t& conv_layer_t::save_grad(ovectorizer_t& s) const
        {
                return s << m_gkdata << m_gwdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ivectorizer_t& conv_layer_t::load_params(ivectorizer_t& s)
        {
                return s >> m_kdata >> m_wdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _forward(
		const tscalar* idata, tsize idims,
                const tscalar* kdata, tsize krows, tsize kcols,
                const tscalar* wdata,
                tscalar* odata, tsize odims, tsize orows, tsize ocols)
        {
		const tsize irows = orows + krows - 1;
                const tsize icols = ocols + kcols - 1;
		const tsize isize = irows * icols;

		const tsize osize = orows * ocols;
		const tsize ksize = krows * kcols;

		for (tsize o = 0; o < odims; o ++)
		{
			tscalar* podata = odata + o * osize;

                	for (tsize r = 0; r < orows; r ++)
	                {	
        	               	for (tsize c = 0; c < ocols; c ++)
				{
					podata[r * ocols + c] = 0;
				}
			}

			for (tsize i = 0; i < idims; i ++)
			{
				const tscalar* pidata = idata + i * isize;
				const tscalar w = wdata[o * idims + i];

	                	for (tsize r = 0; r < orows; r ++)
		                {	
        		               	for (tsize c = 0; c < ocols; c ++)
                		        {
						const tscalar* pkdata = kdata + o * ksize;

                        		        tscalar sum = 0;
                                		for (tsize kr = 0; kr < krows; kr ++)
		                                {
        		                                for (tsize kc = 0; kc < kcols; kc ++)
                		                        {
                        		                        const tscalar iv = pidata[(r + kr) * icols + (c + kc)];
                                		                const tscalar kv = pkdata[kr * kcols + kc];
	
                	                        	        sum += iv * kv;
        	        	                       	}
                                		}					

	                                	podata[r * ocols + c] += w * sum;
					}
        	                }
			}
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& conv_layer_t::forward(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() == input.rows());
                assert(icols() == input.cols());

                m_idata.copy_from(input);
		
		_forward(m_idata.data(), idims(),
			 m_kdata.data(), krows(), kcols(),
			 m_wdata.data(), 
			 m_odata.data(), odims(), orows(), ocols());

                return m_odata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >                        
        static void _backward(
                const tscalar* odata, tsize orows, tsize ocols, 
                const tscalar* kdata, tsize krows, tsize kcols,
                tscalar w, const tscalar* idata,
                tscalar* gkdata, tscalar& gw, tscalar* gidata)
        {
                const tsize icols = ocols + kcols - 1;                
                                
                for (tsize r = 0; r < orows; r ++)
                {
                        for (tsize c = 0; c < ocols; c ++)
                        {
                                for (tsize kr = 0; kr < krows; kr ++)
                                {
                                        for (tsize kc = 0; kc < kcols; kc ++)
                                        {
                                                const tscalar iv = idata[(r + kr) * icols + (c + kc)];
                                                const tscalar ov = odata[r * ocols + c];
                                                const tscalar kv = kdata[kr * kcols + kc];

                                                gidata[(r + kr) * icols + (c + kc)] += ov * kv * w;
                                                gkdata[kr * kcols + kc] += ov * iv * w;
                                                gw += ov * iv * kv;
                                        }
                                }
                        }               
                }
        }

        
	/////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& conv_layer_t::backward(const tensor_t& gradient)
        {
                assert(odims() == gradient.dims());
                assert(orows() == gradient.rows());
                assert(ocols() == gradient.cols());

                m_gkdata.zero();
                m_gwdata.zero();
                m_gidata.zero();

                for (size_t o = 0; o < odims(); o ++)
                {                        
                        auto odata = gradient.plane_data(o);
                        auto kdata = m_kdata.plane_data(o);
                        auto wdata = m_wdata.plane_data(0);

                        auto gkdata = m_gkdata.plane_data(o);
                        auto gwdata = m_gwdata.plane_data(0);

                        for (size_t i = 0; i < idims(); i ++)
                        {
                                auto idata = m_idata.plane_data(i);
                                auto gidata = m_gidata.plane_data(i);

                                _backward(odata, orows(), ocols(),
                                          kdata, krows(), kcols(),
                                          wdata[o * idims() + i], idata,
                                          gkdata, gwdata[o * idims() + i], 
					  gidata);
                        }
                }

                return m_gidata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}


