#include "model.h"
#include "task.h"
#include "util/logger.h"
#include "io/ibstream.h"
#include "io/obstream.h"
#include "text/to_string.hpp"
#include "text/from_string.hpp"
#include <fstream>

namespace cortex
{
        model_manager_t& get_models()
        {
                return model_manager_t::instance();
        }

        model_t::model_t(const string_t& parameters)
                :       clonable_t<model_t>(parameters),
                        m_rows(0),
                        m_cols(0),
                        m_outputs(0),
                        m_color(color_mode::luma)
        {
        }

        bool model_t::save(const string_t& path) const
        {
                std::ofstream os(path, std::ios::binary | std::ios::out | std::ios::trunc);

                io::obstream_t ob(os);

                // save configuration
                ob.write(m_rows);
                ob.write(m_cols);
                ob.write(m_outputs);
                ob.write(text::to_string(m_color));
                ob.write(m_configuration);

                // save parameters
                vector_t params(psize());
                save_params(params);

                ob.write(psize());
                ob.write(params.data(), psize());

                return os.good();
        }

        bool model_t::load(const string_t& path)
        {
                std::ifstream is(path, std::ios::binary | std::ios::in);

                io::ibstream_t ib(is);

                // read configuration
                ib.read(m_rows);
                ib.read(m_cols);
                ib.read(m_outputs);
                { string_t str; ib.read(str); m_color = text::from_string<color_mode>(str); }
                ib.read(m_configuration);

                // apply configuration
                resize(true);

                // read parameters
                tensor_size_t psize = 0;
                ib.read(psize);
                if (psize != this->psize())
                {
                        return false;
                }

                vector_t params(psize);
                ib.read(params.data(), psize);

                // apply parameters
                return load_params(params) && is;
        }

        const tensor_t& model_t::output(const image_t& image, const rect_t& region) const
        {
                return output(image, region.left(), region.top());
        }

        const tensor_t& model_t::output(const image_t& image, coord_t x, coord_t y) const
        {
                return output(make_input(image, x, y));
        }

        const tensor_t& model_t::output(const vector_t& input) const
        {
                assert(input.size() == isize());

                tensor_t xinput(idims(), irows(), icols());
                xinput.vector() = input;

                return output(xinput);
        }

        tensor_t model_t::make_input(const image_t& image, coord_t x, coord_t y) const
        {
                const auto region = rect_t(x, y, icols(), irows());
                return image.to_tensor(region);
        }

        tensor_t model_t::make_input(const image_t& image, const rect_t& region) const
        {
                return make_input(image, region.left(), region.top());
        }

        tensor_size_t model_t::idims() const
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

        bool model_t::resize(const task_t& task, bool verbose)
        {
                return resize(task.irows(), task.icols(), task.osize(), task.color(), verbose);
        }

        bool model_t::resize(const tensor_size_t rows, const tensor_size_t cols, const tensor_size_t outputs,
                const color_mode color, const bool verbose)
        {
                m_rows = rows;
                m_cols = cols;
                m_outputs = outputs;
                m_color = color;
                resize(verbose);

                if (verbose)
                {
                        log_info() << "model: parameters = " << psize() << ".";
                }

                return true;
        }
}
