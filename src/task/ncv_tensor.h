#ifndef NANOCV_TENSOR_H
#define NANOCV_TENSOR_H

#include "ncv_types.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
	// 
	/////////////////////////////////////////////////////////////////////////////////////////

	typedef matrix<vector_t>	tensor_t;
	typedef matrix<matrix_t>	tensor_kernel_t;

	matrices_t to_matrix(const tensor_t& tensor);
	tensor_t to_tensor(const matrix_t& input);
	tensor_t to_tensor(const matrices_t& inputs);

	bool conv(const tensor_t& input, const tensor_kernel_t& kernel, tensor_t& output);


//        class sampler_t
//        {
//        public:
                
//                // Constructor
//                sampler_t(size_t n_labels = 0)
//                        :       m_values(n_labels, 0.0),
//                                m_probs(n_labels, 0.0),
//                                m_n_samples(0),
//                                m_scounts(n_labels, 0),
//                                m_rgen(0.0, 1.0)
//                {
//                }
                
//                // Reset statistics
//                void clear()
//                {
//                        std::fill(m_values.begin(), m_values.end(), 0.0);
//                        std::fill(m_probs.begin(), m_probs.end(), 0.0);
//                        m_n_samples = 0;
//                }
                
//                // Add new samples
//                void add(index_t label, scalar_t value = 1.0)
//                {
//                        m_values[label] += value;
//                        m_n_samples ++;
//                }
//                void add(const sampler_t& other)
//                {
//                        std::transform(m_values.begin(), m_values.end(), other.m_values.begin(),
//                                       m_values.begin(), std::plus<scalar_t>());
//                        m_n_samples += other.m_n_samples;
//                }
                
//                // Specify the desired number of samples to be selected
//                void norm(size_t n_samples);
                
//                // Sampling: return the number of times a sample to be selected
//                size_t select(index_t label, scalar_t value = 1.0) const;
                
//                // Access functions
//                size_t n_labels() const { return m_values.size(); }
//                size_t n_samples() const { return m_n_samples; }
//                scalar_t value(index_t label) const { return m_values[label]; }
//                scalar_t prob(index_t label) const { return m_probs[label]; }
//                size_t scount(index_t label) const { return m_scounts[label]; }
                
//        private:
                
//                // Attributes
//                scalars_t                       m_values;               // Cumulated values / label
//                scalars_t                       m_probs;                // Base probability / label
//                size_t                          m_n_samples;            // #samples (total)
//                mutable counts_t                m_scounts;              // #selected samples / label
//                mutable random_t<scalar_t>      m_rgen;                 // Random number generator (for sampling)
//        };
}

#endif // NANOCV_TENSOR_H
