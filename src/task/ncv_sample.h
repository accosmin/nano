#ifndef NANOCV_SAMPLE_H
#define NANOCV_SAMPLE_H

#include "ncv_types.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // image-indexed sample
        ////////////////////////////////////////////////////////////////////////////////

        struct image_sample_t
        {
                // constructor
                image_sample_t(index_t image = 0, index_t annotation = 0)
                        :       m_image(image), m_annotation(annotation)
                {
                }

                // attributes
                index_t         m_image;        // image index
                index_t         m_annotation;   // annotation index
        };

        typedef std::vector<image_sample_t>     image_samples_t;

        // construct image-indexed samples: [istart, istart + icount) = annotation
        image_samples_t make_image_samples(index_t istart, index_t icount, index_t annotation);

        // fold image-indexed samples
        typedef std::pair<index_t, protocol>    fold_t;
        typedef std::map<
                fold_t,
                image_samples_t>                fold_image_samples_t;

        ////////////////////////////////////////////////////////////////////////////////
        // sample data
        ////////////////////////////////////////////////////////////////////////////////

        struct annotated_image_t;

        struct sample_t
        {
                // check if annotated
                bool has_annotation() const { return m_target.size() > 0; }

                // load from grayscale image
                void load_gray(const annotated_image_t& aimage, const image_sample_t& isample);

                // load from color image
                void load_rgba(const annotated_image_t& aimage, const image_sample_t& isample);

                // attributes
                vector_t        m_data;
                vector_t        m_target;       // target vector (if empty then no annotation is given)
        };

        typedef std::vector<sample_t>           samples_t;
}

#endif // NANOCV_SAMPLE_H
