#include "loss_sparse.h"
#include "text.h"
#include "common/math.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        sparse_loss_t::sparse_loss_t(const string_t& params)
                :       loss_t(params, "sparse-output loss, parameters: w = 1[0,1000]"),
			m_weight(math::clamp(text::from_params<scalar_t>(parameters(), "w", 1.0), 0.0, 1000.0))
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t sparse_loss_t::value(const vector_t&, const vector_t& scores) const
        {
                scalar_t value = 0.0;
                for (auto o = 0; o < scores.rows(); o ++)
                {
			const scalar_t x = scores[o];
			value += math::square(1.0 - x * x) + 	// close to -1/+1
				 m_weight * x * x;		// close to 0
                }

                return 0.5 * value;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        
        vector_t sparse_loss_t::vgrad(const vector_t&, const vector_t& scores) const
        {
                vector_t grads(scores.rows());
                for (auto o = 0; o < scores.rows(); o ++)
                {
			const scalar_t x = scores[o];			
			grads[o] = x * (1.0 - x * x + m_weight);
                }

                return grads;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
