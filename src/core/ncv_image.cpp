#include "ncv_image.h"
#include <SFML/Graphics.hpp>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        bool load_rgba(const string_t& path, rgba_matrix_t& rgba)
        {
                sf::Image image;
                if (!image.loadFromFile(path))
                {
                        return false;
                }

                const int rows = image.getSize().y;
                const int cols = image.getSize().x;

                rgba.resize(rows, cols);
                for (int r = 0; r < rows; r ++)
                {
                        for (int c = 0; c < cols; c ++)
                        {
                                const sf::Color color = image.getPixel(c, r);
                                rgba(r, c) = color::make_rgba(color.r, color.g, color.b, color.a);
                        }
                }

                return true;
        }

        //-------------------------------------------------------------------------------------------------

        bool save_rgba(const string_t& path, const rgba_matrix_t& rgba)
        {
                const int rows = math::cast<int>(rgba.rows());
                const int cols = math::cast<int>(rgba.cols());

                sf::Image image;
                image.create(static_cast<unsigned int>(cols),
                             static_cast<unsigned int>(rows));

                for (int r = 0; r < rows; r ++)
                {
                        for (int c = 0; c < cols; c ++)
                        {
                                const rgba_t color = rgba(r, c);
                                image.setPixel(c, r, sf::Color(
                                        color::make_red(color),
                                        color::make_green(color),
                                        color::make_blue(color),
                                        color::make_alpha(color)));
                        }
                }

                return image.saveToFile(path);
        }

        //-------------------------------------------------------------------------------------------------

        bool image_t::load(const string_t& path)
        {
                return ncv::load_rgba(path, m_rgba);
        }

        //-------------------------------------------------------------------------------------------------

        bool image_t::load_gray(const char* buffer, size_t rows, size_t cols)
        {
                m_rgba.resize(rows, cols);

                for (size_t y = 0, i = 0; y < rows; y ++)
                {
                        for (size_t x = 0; x < cols; x ++, i ++)
                        {
                                m_rgba(y, x) = color::make_rgba(buffer[i], buffer[i], buffer[i]);
                        }
                }

                return true;
        }

        //-------------------------------------------------------------------------------------------------

        bool image_t::load_rgba(const char* buffer, size_t rows, size_t cols)
        {
                m_rgba.resize(rows, cols);

                const size_t size = rows * cols;
                for (size_t y = 0, dr = 0, dg = dr + size, db = dg + size; y < rows; y ++)
                {
                        for (size_t x = 0; x < cols; x ++, dr ++, dg ++, db ++)
                        {
                                m_rgba(y, x) = color::make_rgba(buffer[dr], buffer[dg], buffer[db]);
                        }
                }

                return true;
        }

        //-------------------------------------------------------------------------------------------------
}
