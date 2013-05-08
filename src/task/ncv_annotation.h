#ifndef NANOCV_ANNOTATION_H
#define NANOCV_ANNOTATION_H

#include "ncv_color.h"
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/box.hpp>

namespace ncv
{
        // pixel geometry
        typedef size_t                                          icoord_t;
        typedef boost::geometry::model::d2::point_xy<icoord_t>  ipoint_t;
        typedef boost::geometry::model::box<ipoint_t>           irect_t;

        inline irect_t make_rect(icoord_t x = 0, icoord_t y = 0, icoord_t w = 0, icoord_t h = 0)
        {
                return irect_t(ipoint_t(x, y), ipoint_t(x + w, y + h));
        }

        ////////////////////////////////////////////////////////////////////////////////
        // image annotation
        ////////////////////////////////////////////////////////////////////////////////

        struct annotation_t
        {
                // constructor
                annotation_t(icoord_t x = 0, icoord_t y = 0, icoord_t w = 0, icoord_t h = 0,
                           const string_t& label = string_t(),
                           const vector_t& target = vector_t())
                        :       annotation_t(make_rect(x, y, w, h), label, target)
                {
                }

                annotation_t(const irect_t& region, const string_t& label, const vector_t& target)
                        :       m_region(region),
                                m_label(label),
                                m_target(target)
                {
                }

                // attributes
                irect_t         m_region;       // 2D annotated region
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
                void save_gray(const irect_t& region, vector_t& data) const;
                void save_rgba(const irect_t& region, vector_t& data) const;

                // attributes
                rgba_matrix_t   m_image;
                annotations_t   m_annotations;
                protocol        m_protocol;
        };

        typedef std::vector<annotated_image_t>  annotated_images_t;
}

#endif // NANOCV_ANNOTATION_H
