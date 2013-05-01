#include "ncv_image.h"
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        bool load_image(const string_t& path, rgba_matrix_t& rgba)
        {
                const osg::ref_ptr<osg::Image> image = osgDB::readImageFile(path);
                if (!image)
                {
                        return false;
                }

                const int rows = image->t();
                const int cols = image->s();
                rgba.resize(rows, cols);

                const int channels = image->getImageSizeInBytes() / rows / cols;
                switch (channels)
                {
                case 1:
                        for (int r = 0; r < rows; r ++)
                        {
                                for (int c = 0; c < cols; c ++)
                                {
                                        const unsigned char* data = image->data(c, rows - r - 1);
                                        rgba(r, c) = color::make_rgba(data[0], data[0], data[0], 255);
                                }
                        }
                        return true;

                case 3:
                        for (int r = 0; r < rows; r ++)
                        {
                                for (int c = 0; c < cols; c ++)
                                {
                                        const unsigned char* data = image->data(c, rows - r - 1);
                                        rgba(r, c) = color::make_rgba(data[0], data[1], data[2], 255);
                                }
                        }
                        return true;

                case 4:
                        for (int r = 0; r < rows; r ++)
                        {
                                for (int c = 0; c < cols; c ++)
                                {
                                        const unsigned char* data = image->data(c, rows - r - 1);
                                        rgba(r, c) = color::make_rgba(data[0], data[1], data[2], data[3]);
                                }
                        }
                        return true;

                default:
                        return false;
                }
        }

        //-------------------------------------------------------------------------------------------------

        bool save_image(const string_t& path, const rgba_matrix_t& rgba)
        {
                const int rows = math::cast<int>(rgba.rows());
                const int cols = math::cast<int>(rgba.cols());

                const osg::ref_ptr<osg::Image> image = new osg::Image;

                // RGBA
                if (text::iends_with(path, ".png"))
                {
                        image->allocateImage(cols, rows, 1, GL_RGBA, GL_UNSIGNED_BYTE);

                        for (int r = 0; r < rows; r ++)
                        {
                                for (int c = 0; c < cols; c ++)
                                {
                                        const rgba_t color = rgba(r, c);
                                        unsigned char* data = image->data(c, rows - r - 1);
                                        data[0] = color::make_red(color);
                                        data[1] = color::make_green(color);
                                        data[2] = color::make_blue(color);
                                        data[3] = color::make_alpha(color);
                                }
                        }
                }

                // RGB
                else
                {
                        image->allocateImage(cols, rows, 1, GL_RGB, GL_UNSIGNED_BYTE);
                        for (int r = 0; r < rows; r ++)
                        {
                                for (int c = 0; c < cols; c ++)
                                {
                                        const rgba_t color = rgba(r, c);
                                        unsigned char* data = image->data(c, rows - r - 1);
                                        data[0] = color::make_red(color);
                                        data[1] = color::make_green(color);
                                        data[2] = color::make_blue(color);
                                }
                        }
                }

                return osgDB::writeImageFile(*image, path);
        }

        //-------------------------------------------------------------------------------------------------
}
