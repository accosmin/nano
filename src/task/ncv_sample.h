#ifndef NANOCV_SAMPLE_H
#define NANOCV_SAMPLE_H

#include "ncv_types.h"

namespace ncv
{
        // sample patch
        struct sample
        {
                // check if annotated
                bool has_annotation() const { return m_target.size() > 0; }

                // attributes
                vector_t        m_data;
                vector_t        m_target;       // target vector (if empty then no annotation is given)
        };

        typedef std::vector<sample>             samples_t;
}

#endif // NANOCV_SAMPLE_H
