#ifndef NANOCV_DATASET_H
#define NANOCV_DATASET_H

#include "ncv_tensor.h"

namespace ncv
{
        struct sample;
        typedef std::vector<sample>     samples_t;

        ////////////////////////////////////////////////////////////////////////////////
        // data sample consisting of:
        //      - input 3D tensor (row, cols, feature vector/scalar - e.g. color channels)
        //      - target vector (if annotated)
        //      - label (if annotated)
        //      - scalar weight
        ////////////////////////////////////////////////////////////////////////////////

        struct sample
        {
                // constructor
                sample(const string_t& label = invalid_label(),
                       scalar_t weight = 1.0)
                        :       m_label(label),
                                m_weight(weight)
                {
                }

                // serialize
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
                        ar & m_weight;
                }

                // virtual label for the samples not annotated
                static string_t invalid_label() { return "unknown"; }

                // check if the sample is annotated
                bool annotated() const
                {
                        return m_target.size() > 0 && !m_label.empty() && m_label != invalid_label();
                }

                // attributes
                tensor_data_t   m_input;        // input tensor data
                vector_t        m_target;       // prediction target (if available)
                string_t        m_label;        // label (if available)
                scalar_t        m_weight;
        };
}

#endif // NANOCV_DATASET_H
