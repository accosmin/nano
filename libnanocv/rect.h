#pragma once

#include "point.h"

namespace ncv
{
        ///
        /// \brief 2D rectangle
        ///
        class rect_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit rect_t(coord_t left = 0, coord_t top = 0, coord_t width = 0, coord_t height = 0)
                        :       m_x(left), m_y(top), m_w(width), m_h(height)
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

                point_t center() const { return point_t((left() + right()) / 2, (top() + bottom()) / 2); }

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

        ///
        /// \brief compare two rectangles
        ///
        bool operator==(const rect_t& rect1, const rect_t& rect2);

        ///
        /// \brief stream a rectangle
        ///
        std::ostream& operator<<(std::ostream& s, const rect_t& rect);
}


