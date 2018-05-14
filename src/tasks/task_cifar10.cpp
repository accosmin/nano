#include "logger.h"
#include "task_cifar10.h"
#include "text/algorithm.h"

using namespace nano;

static const string_t tlabels[] =
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
        mem_vision_task_t(tensor3d_dim_t{3, 32, 32}, tensor3d_dim_t{10, 1, 1}, 1),
        m_dir(string_t(std::getenv("HOME")) + "/experiments/databases/cifar10")
{
}

void cifar10_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "dir", m_dir);
}

void cifar10_task_t::to_json(json_t& json) const
{
        nano::to_json(json, "dir", m_dir);
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
                        return load_binary(filename, stream, split2(n_train_samples, protocol::train, 80, protocol::valid));
                }
                else if (nano::iends_with(filename, test_bfile))
                {
                        return load_binary(filename, stream, std::vector<protocol>(n_test_samples, protocol::test));
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

bool cifar10_task_t::load_binary(const string_t& filename, istream_t& stream, const std::vector<protocol>& protocols)
{
        log_info() << "CIFAR-10: loading file <" << filename << "> ...";

        const auto irows = std::get<1>(idims());
        const auto icols = std::get<2>(idims());
        const auto buffer_size = irows * icols * 3;

        std::vector<char> buffer(static_cast<size_t>(buffer_size));
        char label[1];

        size_t icount = 0;
        while ( stream.read(label, 1) == 1 &&
                stream.read(buffer.data(), buffer_size) == buffer_size &&
                icount < protocols.size())
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

                const auto fold = fold_t{0, protocols[icount ++]};
                add_sample(fold, n_chunks() - 1, class_target(ilabel, nano::size(odims())), tlabels[ilabel]);
        }

        // OK
        log_info() << "CIFAR-10: loaded " << icount << " samples.";
        return (protocols.size() == icount);
}
