#include "core/logger.h"
#include "task_cifar10.h"
#include "core/algorithm.h"

using namespace nano;

static const strings_t tlabels =
{
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
};

cifar10_task_t::cifar10_task_t() :
        mem_vision_task_t(make_dims(3, 32, 32), make_dims(10, 1, 1), 10),
        m_dir(string_t(std::getenv("HOME")) + "/experiments/databases/cifar10")
{
}

void cifar10_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "dir", m_dir, "folds", m_folds);
        reconfig(make_dims(3, 32, 32), make_dims(10, 1, 1), m_folds);
}

void cifar10_task_t::to_json(json_t& json) const
{
        nano::to_json(json, "dir", m_dir, "folds", m_folds);
}

bool cifar10_task_t::populate()
{
        const auto bfile = m_dir + "/cifar-10-binary.tar.gz";

        const auto train_bfile1 = "data_batch_1.bin";
        const auto train_bfile2 = "data_batch_2.bin";
        const auto train_bfile3 = "data_batch_3.bin";
        const auto train_bfile4 = "data_batch_4.bin";
        const auto train_bfile5 = "data_batch_5.bin";
        const size_t n_train_samples = 10000;// * 5

        const auto test_bfile = "test_batch.bin";
        const size_t n_test_samples = 10000;

        const auto op = [&] (const string_t& filename, istream_t& stream)
        {
                if (    nano::iends_with(filename, train_bfile1) ||
                        nano::iends_with(filename, train_bfile2) ||
                        nano::iends_with(filename, train_bfile3) ||
                        nano::iends_with(filename, train_bfile4) ||
                        nano::iends_with(filename, train_bfile5))
                {
                        return load_binary(filename, stream, protocol::train, n_train_samples);
                }
                else if (nano::iends_with(filename, test_bfile))
                {
                        return load_binary(filename, stream, protocol::test, n_test_samples);
                }
                else
                {
                        return true;
                }
        };
        const auto error_op = [&] (const string_t& message)
        {
                log_error() << "CIFAR-10: " << message;
        };

        log_info() << "CIFAR-10: loading file <" << bfile << "> ...";

        return nano::load_archive(bfile, op, error_op);
}

bool cifar10_task_t::load_binary(const string_t& filename, istream_t& stream, const protocol p, const size_t count)
{
        log_info() << "CIFAR-10: loading file <" << filename << "> ...";

        const auto chunk_begin = n_chunks();
        const auto sample_begin = size();

        const auto irows = std::get<1>(idims());
        const auto icols = std::get<2>(idims());
        const auto buffer_size = irows * icols * 3;

        std::vector<char> buffer(static_cast<size_t>(buffer_size));
        char label[1];

        // generate samples
        std::vector<tensor_size_t> ilabels;
        while ( stream.read(label, 1) == 1 &&
                stream.read(buffer.data(), buffer_size) == buffer_size &&
                n_chunks() < chunk_begin + count)
        {
                const tensor_size_t ilabel = label[0];
                if (ilabel < 0 || ilabel >= nano::size(odims()))
                {
                        log_error() << "CIFAR-10: invalid label!";
                        return false;
                }

                image_t image;
                image.load_rgb(buffer.data(), irows, icols, irows * icols);
                add_chunk(image, image.hash());

                ilabels.push_back(ilabel);
        }
        if (ilabels.size() != count)
        {
                log_error() << "CIFAR-10: invalid number of samples!";
                return false;
        }

        // generate folds
        add_samples(p, ilabels, tlabels);

        // OK
        log_info() << "CIFAR-10: loaded " << (n_chunks() - chunk_begin) << " samples.";
        return  n_chunks() == chunk_begin + count &&
                size() == sample_begin + count * m_folds;
}
