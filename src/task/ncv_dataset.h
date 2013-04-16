#ifndef NANOCV_DATASET_H
#define NANOCV_DATASET_H

#include "ncv_types.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // dataset where the feature values and the targets are stored in memory.
        //      - data stored as [n_samples() x [input: n_inputs() x 1]]
        //      - targets stored as [n_samples() x [target: n_outputs() x 1]]
        //      - if a target is not provided the samples is flaged as not annotated
        ////////////////////////////////////////////////////////////////////////////////
	
        class dataset
        {
        public:

                // constructor
                dataset();

                // remove all samples
                void clear();

                // add new samples
                bool add(const matrix_t& input, const vector_t& target, const string_t& label);
                bool add(const matrix_t& input, const vector_t& target);
                bool add(const matrix_t& input);
		
                // save/load from file
                bool save(const string_t& path) const;
                bool load(const string_t& path);
                
                // access functions
                bool empty() const { return m_samples.empty(); }
                bool valid() const { return !empty() && irows() > 0 && icols() > 0 && tsize() > 0; }

                size_t size() const { return m_samples.size(); }
                size_t irows() const { return m_irows; }
                size_t icols() const { return m_icols; }
                size_t tsize() const { return m_tsize; }
                
                const matrix_t& input(index_t s) const { return m_samples[s].m_input; }
                const vector_t& target(index_t s) const { return m_samples[s].m_target; }
                const string_t& label(index_t s) const { return m_samples[s].m_label; }

                bool has_input(index_t s) const;
                bool has_target(index_t s) const;
                bool has_label(index_t s) const;

        private:

                // data sample
                struct sample
                {
                        friend class boost::serialization::access;
                        template
                        <
                                class tarchive
                        >
                        void serialize(tarchive& ar, unsigned int /*version*/)
                        {
                                ar & m_input;
                                ar & m_target;
                                ar & m_label;
                        }

                        matrix_t        m_input;        // input data
                        vector_t        m_target;       // prediction target (if available)
                        string_t        m_label;        // label (if available)
                };
		
        private:
		
                // attributes
                std::vector<sample>     m_samples;      // samples
                size_t                  m_irows;        // #input rows
                size_t                  m_icols;        // #input cols
                size_t                  m_tsize;        // #targets
        };
}

#endif // NANOCV_DATASET_H
