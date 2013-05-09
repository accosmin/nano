#ifndef  NANOCV_IMAGE_H
#define  NANOCV_IMAGE_H

#include "ncv_color.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // Save/load RGBA image to disk
        ////////////////////////////////////////////////////////////////////////////////

        bool save_rgba(const string_t& path, const rgba_matrix_t& rgba);
        bool load_rgba(const string_t& path, rgba_matrix_t& rgba);

        ////////////////////////////////////////////////////////////////////////////////
        // image annotation
        ////////////////////////////////////////////////////////////////////////////////

        struct annotation_t
        {
                // constructor
                annotation_t(coord_t x = 0, coord_t y = 0, coord_t w = 0, coord_t h = 0,
                           const string_t& label = string_t(),
                           const vector_t& target = vector_t())
                        :       annotation_t(make_rect(x, y, w, h), label, target)
                {
                }

                annotation_t(const rect_t& region, const string_t& label, const vector_t& target)
                        :       m_region(region),
                                m_label(label),
                                m_target(target)
                {
                }

                // attributes
                rect_t          m_region;       // 2D annotated region
                string_t        m_label;        // label (e.g. classification)
                vector_t        m_target;       // target vector to predict
        };

        typedef std::vector<annotation_t>       annotations_t;

        ////////////////////////////////////////////////////////////////////////////////
        // image with its annotations
        ////////////////////////////////////////////////////////////////////////////////

        struct image_t
        {
                // load gray/color image
                bool load(const string_t& path);
                bool load_gray(const char* buffer, size_t rows, size_t cols);
                bool load_rgba(const char* buffer, size_t rows, size_t cols);

                // attributes
                rgba_matrix_t   m_rgba;
                annotations_t   m_annotations;
                protocol        m_protocol;
        };

        typedef std::vector<image_t>            images_t;
}

#endif //  NANOCV_IMAGE_H
