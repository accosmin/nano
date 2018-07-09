#include "logger.h"
#include "core/mat5.h"
#include "task_svhn.h"
#include "core/color.h"
#include "core/random.h"

using namespace nano;

static const strings_t tlabels =
{
        "digit0",
        "digit1",
        "digit2",
        "digit3",
        "digit4",
        "digit5",
        "digit6",
        "digit7",
        "digit8",
        "digit9"
};

svhn_task_t::svhn_task_t() :
        mem_vision_task_t(make_dims(3, 32, 32), make_dims(10, 1, 1), 10),
        m_dir(string_t(std::getenv("HOME")) + "/experiments/databases/svhn")
{
}

void svhn_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "dir", m_dir, "folds", m_folds);
        reconfig(make_dims(3, 32, 32), make_dims(10, 1, 1), m_folds);
}

void svhn_task_t::to_json(json_t& json) const
{
        nano::to_json(json, "dir", m_dir, "folds", m_folds);
}

bool svhn_task_t::populate()
{
        const auto train_file = m_dir + "/train_32x32.mat";
        const auto extra_file = m_dir + "/extra_32x32.mat";
        const auto test_file = m_dir + "/test_32x32.mat";

        const auto train_count = size_t(73257);
        const auto extra_count = size_t(531131);
        const auto test_count = size_t(26032);

        reserve_chunks(train_count + extra_count + test_count);

        return  load_binary(train_file, protocol::train, train_count) &&
                load_binary(extra_file, protocol::train, extra_count) &&
                load_binary(test_file, protocol::test, test_count);
}

bool svhn_task_t::load_binary(const string_t& path, const protocol p, const size_t count)
{
        log_info() << "SVHN: processing file <" << path << "> ...";

        const auto ecallback = [&] (const string_t& message)
        {
                log_error() << "SVHN: " << message;
        };

        // load header
        const auto hcallback = [&] (const mat5_header_t& header)
        {
                log_info() << "SVHN: header <" << header.description() << ">.";
                return true;
        };

        // load section
        dims_t dims;
        string_t name;

        int section_index = 0;
        const auto scallback = [&] (const mat5_section_t& section, istream_t& stream)
        {
                log_info() << "SVHN: section <" << section << ">.";
                switch (section_index ++)
                {
                // pixels matrix section: {meta, dimensions, name, data} sub-elements
                case 0:         return section.matrix_meta(stream);
                case 1:         return section.matrix_dims(stream, dims);
                case 2:         return section.matrix_name(stream, name);
                case 3:         return section.matrix_data(stream) && load_pixels(section, name, dims, count, stream);

                // labels matrix section: {meta, dimensions, name, data} sub-elements
                case 4:         return section.matrix_meta(stream);
                case 5:         return section.matrix_dims(stream, dims);
                case 6:         return section.matrix_name(stream, name);
                case 7:         return section.matrix_data(stream) && load_labels(section, name, dims, p, count, stream);

                default:        log_error() << "SVHN: unexpected section!"; return false;
                }
        };

        return load_mat5(path, hcallback, scallback, ecallback);
}

bool svhn_task_t::load_pixels(const mat5_section_t& section, const string_t& name, const dims_t& dims, const size_t count,
        istream_t& stream)
{
        log_info() << "SVHN: loading images: name = " << name << ", size = " << join(dims, "x", "", "") << "...";

        const auto irows = std::get<1>(idims());
        const auto icols = std::get<2>(idims());
        const auto px = irows * icols;
        const auto ix = irows * icols * 3;

        // check input
        if (    dims.size() != 4 ||
                static_cast<tensor_size_t>(dims[0]) != irows ||
                static_cast<tensor_size_t>(dims[1]) != icols ||
                dims[2] != 3 ||
                dims[3] * ix > section.m_dsize)
        {
                log_error() << "SVHN: invalid image array size!";
                return false;
        }

        if (    section.m_dtype != mat5_dtype::miUINT8)
        {
                log_error() << "SVHN: expecting miUINT8 image array!";
                return false;
        }

        const auto n_samples = dims[3];
        if (static_cast<size_t>(n_samples) != count)
        {
                log_error() << "SVHN: expecting " << count << " vs. " << n_samples << " samples to load!";
                return false;
        }

        // load images
        auto idata = make_buffer(ix);
        for (tensor_size_t i = 0; i < n_samples; ++ i)
        {
                if (stream.read(idata.data(), ix) != ix)
                {
                        log_error() << "SVHN: failed to load image!";
                        return false;
                }

                image_t image(irows, icols, color_mode::rgb);
                image.plane(0) = nano::map_matrix(idata.data() + 0 * px, icols, irows).cast<luma_t>().transpose();
                image.plane(1) = nano::map_matrix(idata.data() + 1 * px, icols, irows).cast<luma_t>().transpose();
                image.plane(2) = nano::map_matrix(idata.data() + 2 * px, icols, irows).cast<luma_t>().transpose();
                add_chunk(image, image.hash());
        }

        return stream.skip(section.m_dsize - n_samples * ix);
}

bool svhn_task_t::load_labels(const mat5_section_t& section,
        const string_t& name, const dims_t& dims, const protocol p, const size_t count,
        istream_t& stream)
{
        log_info() << "SVHN: loading labels: name = " << name << ", size = " << join(dims, "x", "", "") << "...";

        // check input
        if (    dims.size() != 2 ||
                dims[1] != 1 ||
                dims[0] > section.m_dsize)
        {
                log_error() << "SVHN: invalid label array size!";
                return false;
        }

        if (    section.m_dtype != nano::mat5_dtype::miUINT8)
        {
                log_error() << "SVHN: expecting miUINT8 label array!";
                return false;
        }

        const auto n_samples = dims[0];
        if (static_cast<size_t>(n_samples) != count)
        {
                log_error() << "SVHN: expecting " << count << " vs. " << n_samples << " samples to load!";
                return false;
        }

        // load labels
        std::vector<tensor_size_t> ilabels;
        ilabels.reserve(count);
        for (size_t i = 0; i < count; ++ i)
        {
                char ldata;
                if (!stream.read(ldata))
                {
                        log_error() << "SVHN: failed to load label!";
                        return false;
                }

                tensor_size_t ilabel = ldata;
                if (ilabel == 10)
                {
                        ilabel = 0;
                }
                else if (ilabel < 1 || ilabel > 9)
                {
                        log_error() << "SVHN: invalid label <" << ilabel << ">!";
                        return false;
                }

                ilabels.push_back(ilabel);
        }

        // generage folds
        add_samples(p, ilabels, tlabels);
        return stream.skip(section.m_dsize - n_samples);
}
