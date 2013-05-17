#ifndef NANOCV_SAMPLE_H
#define NANOCV_SAMPLE_H

#include "ncv_image.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // image-indexed sample
        ////////////////////////////////////////////////////////////////////////////////

        struct isample_t
        {
                // constructor
                isample_t(size_t index = 0, coord_t x = 0, coord_t y = 0, coord_t w = 0, coord_t h = 0)
                        :       isample_t(index, geom::make_rect(x, y, w, h))
                {
                }
                isample_t(size_t index, const rect_t& region)
                        :       m_index(index), m_region(region)
                {
                }

                // attributes
                size_t          m_index;        // image index
                rect_t          m_region;       // image coordinates
        };

        typedef std::vector<isample_t>          isamples_t;

        // fold image-indexed samples
        typedef std::pair<size_t, protocol>     fold_t;
        typedef std::map<fold_t, isamples_t>    folds_t;

        ////////////////////////////////////////////////////////////////////////////////
        // sample data
        ////////////////////////////////////////////////////////////////////////////////

        struct sample_t
        {
                // check if annotated
                bool has_annotation() const { return m_target.size() > 0; }

                // load from gray/color image
                void load_gray(const image_t& image, const rect_t& region);
                void load_rgba(const image_t& image, const rect_t& region);

                // load target
                void load_target(const image_t& image, const rect_t& region);

                // attributes
                vector_t        m_input;
                vector_t        m_target;       // target vector (if empty then no annotation is given)
        };

        typedef std::vector<sample_t>           samples_t;
}

#endif // NANOCV_SAMPLE_H
