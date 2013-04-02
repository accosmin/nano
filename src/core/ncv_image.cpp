#include "ncv_image.h"
#include <QImage>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        image::image(size_t rows, size_t cols, const string_t& name)
                :       m_rgba(rows, cols),
                        m_name(name)
        {
        }

        //-------------------------------------------------------------------------------------------------

        image::image(const rgba_matrix_t& rgba, const string_t& name)
                :       m_rgba(rgba),
                        m_name(name)
        {
        }

        //-------------------------------------------------------------------------------------------------

        bool image::load(const rgba_matrix_t& rgba)
        {
                m_rgba = rgba;
                return false == empty();
        }

        //-------------------------------------------------------------------------------------------------

        bool image::load(const string_t& path)
        {
                QImage image;
                if (!image.load(path.c_str()))
                {
                        return false;
                }
                
                const int rows = image.height();
                const int cols = image.width();
                
                m_rgba.resize(rows, cols);                
                for (int r = 0; r < rows; r ++)
                {
                        for (int c = 0; c < cols; c ++)
                        {
                                const QRgb color = image.pixel(c, r);
                                m_rgba(r, c) = color::rgba(qRed(color), qGreen(color), qBlue(color), qAlpha(color));
                        }
                }
                
                return false == empty();
        }

        //-------------------------------------------------------------------------------------------------

        bool image::save(const string_t& path) const
        {
                QImage image(math::cast<int>(cols()),
                             math::cast<int>(rows()),
                             QImage::Format_RGB32);

                for (size_t r = 0; r < rows(); r ++)
                {
                        for (size_t c = 0; c < cols(); c ++)
                        {
                                const rgba_t color = m_rgba(r, c);
                                image.setPixel(c, r, qRgba(
                                        color::red(color),
                                        color::green(color),
                                        color::blue(color),
                                        color::alpha(color)));
                        }
                }
                
                return image.save(path.c_str());
        }

        //-------------------------------------------------------------------------------------------------

        bool image::save(const string_t& path, channel ch) const
        {
                QImage image(math::cast<int>(cols()),
                             math::cast<int>(rows()),
                             QImage::Format_RGB32);

                switch (ch)
                {
                case channel::red:
                        for (size_t r = 0; r < rows(); r ++)
                        {
                                for (size_t c = 0; c < cols(); c ++)
                                {
                                        const rgba_t color = m_rgba(r, c);
                                        image.setPixel(c, r, qRgba(color::red(color), 255, 255, 255));
                                }
                        }
                        break;

                case channel::green:
                        for (size_t r = 0; r < rows(); r ++)
                        {
                                for (size_t c = 0; c < cols(); c ++)
                                {
                                        const rgba_t color = m_rgba(r, c);
                                        image.setPixel(c, r, qRgba(255, color::green(color), 255, 255));
                                }
                        }
                        break;

                case channel::blue:
                        for (size_t r = 0; r < rows(); r ++)
                        {
                                for (size_t c = 0; c < cols(); c ++)
                                {
                                        const rgba_t color = m_rgba(r, c);
                                        image.setPixel(c, r, qRgba(255, 255, color::blue(color), 255));
                                }
                        }
                        break;

                case channel::luma:
                        for (size_t r = 0; r < rows(); r ++)
                        {
                                for (size_t c = 0; c < cols(); c ++)
                                {
                                        const rgba_t color = m_rgba(r, c);
                                        const rgba_t luma = color::luma(color);
                                        image.setPixel(c, r, qRgba(luma, luma, luma, 255));
                                }
                        }
                        break;
                }
                
                return image.save(path.c_str());
        }

        //-------------------------------------------------------------------------------------------------
}
