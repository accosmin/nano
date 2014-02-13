#include "model.h"
#include "common/logger.h"
#include "task.h"
#include <fstream>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        model_t::model_t()
                :       clonable_t<model_t>(),
                        m_rows(0),
                        m_cols(0),
                        m_outputs(0),
                        m_parameters(0),
                        m_color(color_mode::luma)
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool model_t::save(const string_t& path) const
        {
                std::ofstream os(path, std::ios::binary);

                boost::archive::binary_oarchive oa(os);
                oa << m_rows;
                oa << m_cols;
                oa << m_outputs;
                oa << m_parameters;
                oa << m_color;

                return save(oa) && os.good();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool model_t::load(const string_t& path)
        {
                std::ifstream is(path, std::ios::binary);

                boost::archive::binary_iarchive ia(is);
                ia >> m_rows;
                ia >> m_cols;
                ia >> m_outputs;
                ia >> m_parameters;
                ia >> m_color;

                return load(ia) && is.good();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        vector_t model_t::value(const image_t& image, const rect_t& region) const
        {
                return value(image, geom::left(region), geom::top(region));
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        vector_t model_t::value(const image_t& image, coord_t x, coord_t y) const
        {
                return value(make_input(image, x, y));
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        tensor3d_t model_t::make_input(const image_t& image, coord_t x, coord_t y) const
        {
                tensor3d_t data;

                const rect_t region = geom::make_rect(x, y, n_cols(), n_rows());

                switch (m_color)
                {
                case color_mode::luma:
                        data.resize(1, n_rows(), n_cols());
                        data(0) = image.make_luma(region);
                        break;

                case color_mode::rgba:
                        data.resize(3, n_rows(), n_cols());
                        data(0) = image.make_red(region);
                        data(1) = image.make_green(region);
                        data(2) = image.make_blue(region);
                        break;
                }

                return data;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        tensor3d_t model_t::make_input(const image_t& image, const rect_t& region) const
        {
                return make_input(image, geom::left(region), geom::top(region));
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t model_t::n_inputs() const
        {
                switch (m_color)
                {
                case color_mode::rgba:
                        return 3;

                case color_mode::luma:
                default:
                        return 1;
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool model_t::resize(const task_t& task)
        {
                return resize(task.n_rows(), task.n_cols(), task.n_outputs(), task.color());
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool model_t::resize(size_t rows, size_t cols, size_t outputs, color_mode color)
        {
                m_rows = rows;
                m_cols = cols;
                m_outputs = outputs;
                m_color = color;
                m_parameters = resize();

                log_info() << "model: parameters = " << n_parameters() << ".";

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
