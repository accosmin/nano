#pragma once

#include <cstdint>
#include <ostream>
#include <algorithm>

namespace nano
{
        ///
        /// \brief image coordinate (in pixels) - 64bit to be compatible with Eigen::Index
        ///
        using coord_t = int64_t;

        ///
        /// \brief image area (in pixels)
        ///
        using area_t = int64_t;

        ///
        /// \brief 2D point
        ///
        class point_t
        {
        public:
                ///
                /// \brief constructor
                ///
                explicit point_t(coord_t x = 0, coord_t y = 0) :
                        m_x(x), m_y(y)
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

        inline bool operator==(const point_t& point1, const point_t& point2)
        {
                return  point1.x() == point2.x() &&
                        point1.y() == point2.y();
        }

        inline std::ostream& operator<<(std::ostream& s, const point_t& point)
        {
                return s << "{POINT: (" << point.x() << ", " << point.y() << "}";
        }

        ///
        /// \brief 2D rectangle
        ///
        class rect_t
        {
        public:
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
        /// \brief [0, 100] percentage overlap between two rectangle (aka Jaccard distance)
        ///
        inline area_t overlap(const rect_t& rect1, const rect_t& rect2)
        {
                return 100 * ((rect1 & rect2).area() + 1) / ((rect1 | rect2).area() + 1);
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
