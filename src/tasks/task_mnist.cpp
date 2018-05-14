#include "logger.h"
#include "task_mnist.h"
#include "io/archive.h"

using namespace nano;

static const string_t tlabels[] =
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

mnist_task_t::mnist_task_t() :
        mem_vision_task_t(tensor3d_dim_t{1, 28, 28}, tensor3d_dim_t{10, 1, 1}, 1),
        m_dir(string_t(std::getenv("HOME")) + "/experiments/databases/mnist")
{
}

void mnist_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "dir", m_dir);
}

void mnist_task_t::to_json(json_t& json) const
{
        nano::to_json(json, "dir", m_dir);
}

bool mnist_task_t::populate()
{
        const auto test_ifile = m_dir + "/t10k-images-idx3-ubyte.gz";
        const auto test_gfile = m_dir + "/t10k-labels-idx1-ubyte.gz";
        const auto test_proto = std::vector<protocol>(10000, protocol::test);

        const auto train_ifile = m_dir + "/train-images-idx3-ubyte.gz";
        const auto train_gfile = m_dir + "/train-labels-idx1-ubyte.gz";
        const auto train_proto = split2(60000, protocol::train, 80, protocol::valid);

        return  load_binary(train_ifile, train_gfile, train_proto) &&
                load_binary(test_ifile, test_gfile, test_proto);
}

bool mnist_task_t::load_binary(const string_t& ifile, const string_t& gfile, const std::vector<protocol>& protocols)
{
        size_t iindex = n_chunks();
        size_t icount = 0;
        size_t gcount = 0;

        const auto irows = std::get<1>(idims());
        const auto icols = std::get<2>(idims());
        const auto buffer_size = irows * icols;

        std::vector<char> buffer(static_cast<size_t>(buffer_size));
        char label[2];

        const auto error_op = [&] (const string_t& message)
        {
                log_error() << "MNIST: " << message;
        };

        // load images
        const auto iop = [&] (const string_t&, istream_t& stream)
        {
                if (stream.read(buffer.data(), 16) != 16)
                {
                        return false;
                }

                while (stream.read(buffer.data(), buffer_size) == buffer_size)
                {
                        image_t image;
                        image.load_luma(buffer.data(), irows, icols);
                        add_chunk(image, image.hash());

                        ++ icount;
                }

                return true;
        };

        log_info() << "MNIST: loading file <" << ifile << "> ...";
        if (!nano::load_archive(ifile, iop, error_op))
        {
                log_error() << "MNIST: failed to load file <" << ifile << ">!";
                return false;
        }

        // load ground truth
        const auto gop = [&] (const string_t&, istream_t& stream)
        {
                if (stream.read(buffer.data(), 8) != 8)
                {
                        return false;
                }

                while (stream.read(label, 1) == 1 && gcount < protocols.size())
                {
                        const auto ilabel = static_cast<tensor_size_t>(label[0]);
                        if (ilabel < 0 || ilabel >= nano::size(odims()))
                        {
                                log_error() << "MNIST: invalid label!";
                                return false;
                        }

                        const auto fold = fold_t{0, protocols[gcount]};
                        add_sample(fold, iindex, class_target(ilabel, nano::size(odims())), tlabels[ilabel]);

                        ++ gcount;
                        ++ iindex;
                }

                return true;
        };

        log_info() << "MNIST: loading file <" << gfile << "> ...";
        if (!nano::load_archive(gfile, gop, error_op))
        {
                log_error() << "MNIST: failed to load file <" << gfile << ">!";
                return false;
        }

        // OK
        log_info() << "MNIST: loaded " << icount << "/" << gcount << " samples.";
        return (protocols.size() == gcount) && (protocols.size() == icount);
}
