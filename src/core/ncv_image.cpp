#include "ncv_image.h"
#include <SFML/Graphics.hpp>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        bool load_image(const string_t& path, rgba_matrix_t& rgba)
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

        bool save_image(const string_t& path, const rgba_matrix_t& rgba)
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
}
