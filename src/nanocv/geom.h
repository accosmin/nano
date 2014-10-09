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
        class point_t
        {
        public:

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

        private:

                // attributes
                coord_t         m_x;
                coord_t         m_y;
        };

        ///
        /// \brief 2D rectangle
        ///
        class rect_t
        {
        public:

                ///
                /// \brief constructor
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
                bool empty() const { return area() == 0; }
                bool valid() const { return m_w >= 0 && m_h >= 0; }

        private:

                // attributes
                coord_t         m_x;            ///< left
                coord_t         m_y;            ///< top
                coord_t         m_w;            ///< width
                coord_t         m_h;            ///< height
        };

        ///
        /// \brief intersect two rectangles
        ///
        rect_t operator&(const rect_t& rect1, const rect_t& rect2);

        ///
        /// \brief union with another rectangle
        ///
        rect_t operator|(const rect_t& rect1, const rect_t& rect2);

        ///
        /// \brief [0, 1] overlap between two rectangle (aka Jaccard distance)
        ///
        scalar_t overlap(const rect_t& rect1, const rect_t& rect2);
}


