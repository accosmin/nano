#include "model.h"
#include "task.h"
#include "core/logger.h"
#include <fstream>

namespace ncv
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
                std::ofstream os(path, std::ios::binary);

                boost::archive::binary_oarchive oa(os);
                oa << m_rows;
                oa << m_cols;
                oa << m_outputs;
                oa << m_color;

                return save(oa) && os.good();
        }

        bool model_t::load(const string_t& path)
        {
                std::ifstream is(path, std::ios::binary);

                boost::archive::binary_iarchive ia(is);
                ia >> m_rows;
                ia >> m_cols;
                ia >> m_outputs;
                ia >> m_color;

                return load(ia) && is.good();
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
                assert(static_cast<size_t>(input.size()) == isize());

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

        size_t model_t::idims() const
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

        bool model_t::resize(size_t rows, size_t cols, size_t outputs, color_mode color, bool verbose)
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
