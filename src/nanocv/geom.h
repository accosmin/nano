#pragma once

#include "types.h"

namespace ncv
{
        ///
        /// \brief image coordinate
        ///
        typedef int16_t                 coord_t;
        typedef int32_t                 area_t;

        ///
        /// \brief 2D point
        ///
        struct point_t
        {
                ///
                /// \brief constructor
                ///
                point_t(coord_t x = 0, coord_t y = 0)
                        :       m_x(x), m_y(y)
                {
                }

                ///
                /// \brief access functions
                ///
                coord_t x() const { return m_x; }
                coord_t y() const { return m_y; }

                coord_t& x() { return m_x; }
                coord_t& y() { return m_y; }

                // attributes
                coord_t         m_x;
                coord_t         m_y;
        };

        ///
        /// \brief 2D rectangle
        ///
        struct rect_t
        {
                ///
                /// \brief rect_t
                /// \param x
                /// \param y
                /// \param w
                /// \param h
                ///
                rect_t(coord_t x = 0, coord_t y = 0, coord_t w = 0, coord_t h = 0)
                        :       m_x(x), m_y(y), m_w(w), m_h(h)
                {
                }

                ///
                /// \brief access functions
                ///
                coord_t top() const { return m_y; }
                coord_t left() const { return m_x; }
                coord_t right() const { return left() + width(); }
                coord_t bottom() const { return top() + height(); }

                coord_t width() const { return m_w; }
                coord_t height() const { return m_h; }

                coord_t rows() const { return height(); }
                coord_t cols() const { return width(); }

                area_t area() const { return area_t(width()) * area_t(height()); }

                ///
                /// \brief intersection rectangle
                ///
                rect_t intersection(const rect_t& other) const;

                ///
                /// \brief union with another rectangle
                ///
                rect_t union_(const rect_t& other) const;

                ///
                /// \brief [0, 1] overlap between two rectangle (aka Jaccard distance)
                ///
                scalar_t overlap(const rect_t& other) const;

                // attributes
                coord_t         m_x;            ///< left
                coord_t         m_y;            ///< top
                coord_t         m_w;            ///< width
                coord_t         m_h;            ///< height
        };
}


