#ifndef NANOCV_SAMPLE_H
#define NANOCV_SAMPLE_H

#include "ncv_types.h"

namespace ncv
{
        // image-indexed sample
        struct image_sample_t
        {
                // constructor
                image_sample_t(index_t image = 0, index_t row = 0, index_t col = 0)
                        :       m_image(image), m_row(row), m_col(col)
                {
                }

                // attributes
                index_t         m_image;        // image index
                index_t         m_row, m_col;   // starting row, column in the image
        };

        typedef std::vector<image_sample_t>     image_samples_t;

        // construct image-indexed samples [istart, istart + icount)
        image_samples_t make_image_samples(index_t istart, index_t icount, index_t row = 0, index_t col = 0);

        // fold image-indexed samples
        typedef std::pair<index_t, protocol>    fold_t;
        typedef std::map<
                fold_t,
                image_samples_t>                fold_image_samples_t;

        // sample data
        struct sample_t
        {
                // check if annotated
                bool has_annotation() const { return m_target.size() > 0; }

                // attributes
                vector_t        m_data;
                vector_t        m_target;       // target vector (if empty then no annotation is given)
        };

        typedef std::vector<sample_t>           samples_t;
}

#endif // NANOCV_SAMPLE_H
