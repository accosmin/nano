#pragma once

#include "point.h"
#include "scalar.h"
#include <algorithm>

namespace nano
{
        ///
        /// \brief 2D rectangle
        ///
        struct rect_t
        {
                ///
                /// \brief constructor
                ///
                explicit rect_t(coord_t left = 0, coord_t top = 0, coord_t width = 0, coord_t height = 0) :
                        m_x(left), m_y(top), m_w(width), m_h(height)
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
                bool empty() const { return m_w == 0 || m_h == 0; }
                bool valid() const { return m_w >= 0 && m_h >= 0; }

                point_t center() const { return point_t((left() + right()) / 2, (top() + bottom()) / 2); }

        private:

                // attributes
                coord_t         m_x;            ///< left
                coord_t         m_y;            ///< top
                coord_t         m_w;            ///< width
                coord_t         m_h;            ///< height
        };

        inline std::ostream& operator<<(std::ostream& s, const rect_t& rect)
        {
                return s << "{RECT: top-left = (" << rect.left() << ", " << rect.top()
                         << "), size = " << rect.width() << "x" << rect.height() << "}";
        }

        ///
        /// \brief intersect two rectangles
        ///
        inline rect_t operator&(const rect_t& rect1, const rect_t& rect2)
        {
                const auto top = std::max(rect1.top(), rect2.top());
                const auto left = std::max(rect1.left(), rect2.left());
                const auto right = std::min(rect1.right(), rect2.right());
                const auto bottom = std::min(rect1.bottom(), rect2.bottom());

                return  (right >= left && bottom >= top) ?
                        rect_t(left, top, right - left, bottom - top) :
                        rect_t(0, 0, 0, 0);
        }

        ///
        /// \brief union with another rectangle
        ///
        inline rect_t operator|(const rect_t& rect1, const rect_t& rect2)
        {
                const auto top = std::min(rect1.top(), rect2.top());
                const auto left = std::min(rect1.left(), rect2.left());
                const auto right = std::max(rect1.right(), rect2.right());
                const auto bottom = std::max(rect1.bottom(), rect2.bottom());

                return rect_t(left, top, right - left, bottom - top);
        }

        ///
        /// \brief [0, 1] overlap between two rectangle (aka Jaccard distance)
        ///
        inline scalar_t overlap(const rect_t& rect1, const rect_t& rect2)
        {
                return static_cast<scalar_t>((rect1 & rect2).area() + 1) /
                       static_cast<scalar_t>((rect1 | rect2).area() + 1);
        }

        ///
        /// \brief compare two rectangles
        ///
        inline bool operator==(const rect_t& rect1, const rect_t& rect2)
        {
                return  rect1.left() == rect2.left() &&
                        rect1.top() == rect2.top() &&
                        rect1.width() == rect2.width() &&
                        rect1.height() == rect2.height();
        }
}
