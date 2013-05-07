#ifndef NANOCV_ANNOTATION_H
#define NANOCV_ANNOTATION_H

#include "ncv_color.h"
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/box.hpp>

namespace ncv
{
        // pixel geometry
        typedef int                                             coord_t;
        typedef boost::geometry::model::d2::point_xy<coord_t>   point_t;
        typedef boost::geometry::model::box<point_t>            rect_t;

        ////////////////////////////////////////////////////////////////////////////////
        // image annotation
        ////////////////////////////////////////////////////////////////////////////////

        struct annotation_t
        {
                // constructor
                annotation_t(coord_t x = 0, coord_t y = 0, coord_t w = 0, coord_t h = 0,
                           const string_t& label = string_t(),
                           const vector_t& target = vector_t())
                        :       m_region(point_t(x, y), point_t(x + w, y + h)),
                                m_label(label),
                                m_target(target)
                {
                }

                // attributes
                rect_t          m_region;       // 2D annotated region
                string_t        m_label;        //
                vector_t        m_target;       // target vector
        };

        typedef std::vector<annotation_t>       annotations_t;

        ////////////////////////////////////////////////////////////////////////////////
        // image with its annotations
        ////////////////////////////////////////////////////////////////////////////////

        struct annotated_image_t
        {
                // load gray/color image from buffer
                void load_gray(const char* buffer, size_t rows, size_t cols);
                void load_rgba(const char* buffer, size_t rows, size_t cols);

                // save image region to gray/color buffer
                void save_gray(size_t row, size_t col, size_t rows, size_t cols, vector_t& data) const;
                void save_rgba(size_t row, size_t col, size_t rows, size_t cols, vector_t& data) const;

                // attributes
                rgba_matrix_t   m_image;
                annotations_t   m_annotations;
                protocol        m_protocol;
        };

        typedef std::vector<annotated_image_t>  annotated_images_t;
}

#endif // NANOCV_ANNOTATION_H
