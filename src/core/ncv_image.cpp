#include "ncv_image.h"
#include <QImage>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        bool load_image(const string_t& path, rgba_matrix_t& rgba)
        {
                QImage image;
                if (!image.load(path.c_str()))
                {
                        return false;
                }
                
                const int rows = image.height();
                const int cols = image.width();
                
                rgba.resize(rows, cols);
                for (int r = 0; r < rows; r ++)
                {
                        for (int c = 0; c < cols; c ++)
                        {
                                const QRgb color = image.pixel(c, r);
                                rgba(r, c) = color::encode_rgba(qRed(color), qGreen(color), qBlue(color), qAlpha(color));
                        }
                }
                
                return true;
        }

        //-------------------------------------------------------------------------------------------------

        bool save_image(const string_t& path, const rgba_matrix_t& rgba)
        {
                const int rows = math::cast<int>(rgba.rows());
                const int cols = math::cast<int>(rgba.cols());

                QImage image(cols, rows, QImage::Format_RGB32);

                for (int r = 0; r < rows; r ++)
                {
                        for (int c = 0; c < cols; c ++)
                        {
                                const rgba_t color = rgba(r, c);
                                image.setPixel(c, r, qRgba(
                                        color::decode_red(color),
                                        color::decode_green(color),
                                        color::decode_blue(color),
                                        color::decode_alpha(color)));
                        }
                }
                
                return image.save(path.c_str());
        }

        //-------------------------------------------------------------------------------------------------
}
