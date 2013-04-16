#include "ncv_sampler.h"
//#include "ncv_math.h"

//namespace ncv
//{
//        //-------------------------------------------------------------------------------------------------

//        void sampler_t::norm(size_t n_samples)
//        {
//                for (size_t label = 0; label < m_values.size(); label ++)
//                {
//                        m_probs[label] =        inversem_values[label]) *
//                                                inversem_values.size()) *
//                                                n_samples;
                        
//                        m_scounts[label] = 0;
//                }
//        }

//        //-------------------------------------------------------------------------------------------------

//        size_t sampler_t::select(index_t label, scalar_t value) const
//        {
//                const scalar_t prob = m_probs[label] * value;
                
//                size_t iprob = (size_t)prob;
//                if (m_rgen() < prob - iprob)
//                {
//                        iprob ++;
//                }
                
//                m_scounts[label] += iprob;
//                return iprob;
//        }

//        //-------------------------------------------------------------------------------------------------
//}
