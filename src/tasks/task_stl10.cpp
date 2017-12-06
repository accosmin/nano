#include "logger.h"
#include "task_stl10.h"
#include "text/algorithm.h"

using namespace nano;

static const string_t tlabels[] =
{
        "airplane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
        "ship",
        "truck"
};

stl10_task_t::stl10_task_t() :
        mem_vision_task_t(tensor3d_dim_t{3, 96, 96}, tensor3d_dim_t{10, 1, 1}, 10),
        m_dir(string_t(std::getenv("HOME")) + "/experiments/databases/stl10")
{
}

json_reader_t& stl10_task_t::config(json_reader_t& reader)
{
        return reader.object("dir", m_dir);
}

json_writer_t& stl10_task_t::config(json_writer_t& writer) const
{
        return writer.object("dir", m_dir);
}

bool stl10_task_t::populate()
{
        m_samples.clear();

        const string_t bfile = m_dir + "/stl10_binary.tar.gz";

        const string_t train_ifile = "train_X.bin";
        const string_t train_gfile = "train_y.bin";
        const string_t train_uifile = "unlabeled_X.bin";
        const size_t n_train = 5000;
        const size_t n_unlabeled = 100000;

        const string_t test_ifile = "test_X.bin";
        const string_t test_gfile = "test_y.bin";
        const size_t n_test = 8000;

        const string_t fold_file = "fold_indices.txt";

        const auto op = [&] (const string_t& filename, istream_t& stream)
        {
                if (nano::iends_with(filename, train_ifile))
                {
                        return load_ifile(filename, stream, false, n_train);
                }
                else if (nano::iends_with(filename, train_gfile))
                {
                        return load_gfile(filename, stream, n_train);
                }
                else if (nano::iends_with(filename, test_ifile))
                {
                        return load_ifile(filename, stream, false, n_test);
                }
                else if (nano::iends_with(filename, test_gfile))
                {
                        return load_gfile(filename, stream, n_test);
                }
                else if (nano::iends_with(filename, train_uifile))
                {
                        return load_ifile(filename, stream, true, n_unlabeled);
                }
                else if (nano::iends_with(filename, fold_file))
                {
                        return load_folds(filename, stream, n_test, n_train, n_unlabeled);
                }
                else
                {
                        return true;
                }
        };
        const auto error_op = [&] (const string_t& message)
        {
                log_error() << "STL-10: " << message;
        };

        log_info() << "STL-10: loading file <" << bfile << "> ...";

        return nano::load_archive(bfile, op, error_op);
}

bool stl10_task_t::load_ifile(const string_t& ifile, istream_t& stream,
        const bool unlabeled, const size_t count)
{
        log_info() << "STL-10: loading file <" << ifile << "> ...";

        const auto irows = std::get<1>(idims());
        const auto icols = std::get<2>(idims());
        const auto px = irows * icols;
        const auto ix = 3 * px;

        std::vector<char> buffer(static_cast<size_t>(ix));
        auto iptr = buffer.data();

        size_t icount = 0;

        // load images
        while (stream.read(buffer.data(), ix) == ix)
        {
                image_t image(irows, icols, color_mode::rgb);
                image.plane(0) = nano::map_matrix(iptr + 0 * px, icols, irows).cast<luma_t>().transpose();
                image.plane(1) = nano::map_matrix(iptr + 1 * px, icols, irows).cast<luma_t>().transpose();
                image.plane(2) = nano::map_matrix(iptr + 2 * px, icols, irows).cast<luma_t>().transpose();
                add_chunk(image, image.hash());

                if (unlabeled)
                {
                        m_samples.emplace_back(n_chunks() - 1, nano::size(odims()));
                }

                ++ icount;
        }

        log_info() << "STL-10: loaded " << icount << " images.";

        return count == icount;
}

bool stl10_task_t::load_gfile(const string_t& gfile, istream_t& stream,
        const size_t count)
{
        log_info() << "STL-10: loading file <" << gfile << "> ...";

        char label;

        size_t iindex = n_chunks() - count;
        size_t gcount = 0;

        // load annotations
        while (stream.read(&label, 1) == 1)
        {
                const auto ilabel = static_cast<tensor_size_t>(label) - 1;

                if (ilabel >= 0 && ilabel < nano::size(odims()))
                {
                        m_samples.emplace_back(iindex, ilabel);
                }
                else
                {
                        m_samples.emplace_back(iindex, nano::size(odims()));
                }

                ++ gcount;
                ++ iindex;
        }

        log_info() << "STL-10: loaded " << gcount << " samples.";

        return count == gcount;
}

bool stl10_task_t::load_folds(const string_t& ifile, istream_t& stream,
        const size_t n_test, const size_t n_train, const size_t n_unlabeled)
{
        log_info() << "STL-10: loading file <" << ifile << "> ...";

        // NB: samples arranged like [n_test][n_train][n_unlabeled]
        const auto orig_samples = m_samples;

        const auto op_sample = [&] (const fold_t& fold, const sample_t& sample)
        {
                if (sample.m_label < nano::size(odims()))
                {
                        add_sample(fold, sample.m_image,
                                   class_target(sample.m_label, nano::size(odims())),
                                   tlabels[sample.m_label]);
                }
                else
                {
                        add_sample(fold, sample.m_image);
                }
        };

        const size_t fold_size = 1000;

        // training samples [0, n_train) ...
        for (size_t f = 0; f < fsize(); ++ f)
        {
                string_t line;
                if (!stream.getline(line))
                {
                        return false;
                }

                const auto tokens = nano::split(line, " \t\n\r");

                size_t fcount = 0;
                for (const auto& token : tokens)
                {
                        const auto i = nano::from_string<size_t>(token);
                        if (i < n_train)
                        {
                                op_sample(make_fold(f, protocol::train), m_samples[n_test + i]);
                                ++ fcount;
                        }
                        else
                        {
                                return false;
                        }
                }

                log_info() << "STL-10: loaded " << fcount << "/" << tokens.size()
                           << " samples for fold [" << (f + 1) << "/" << fsize() << "].";
                if (fcount != fold_size)
                {
                        return false;
                }
        }

        // unlabeled samples [n_train, n_train + n_unlabeled)
        for (size_t f = 0; f < fsize(); ++ f)
        {
                for (size_t i = 0; i < n_unlabeled; ++ i)
                {
                        op_sample(make_fold(f, protocol::train), m_samples[n_test + n_train + i]);
                }
        }

        // testing samples [n_train + n_unlabeled, n_train + n_unlabeled + n_test)
        for (size_t f = 0; f < fsize(); ++ f)
        {
                for (size_t i = 0; i < n_test; ++ i)
                {
                        op_sample(make_fold(f, protocol::test), m_samples[i]);
                }
        }

        return true;
}
