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
        mem_vision_task_t(make_dims(1, 28, 28), make_dims(10, 1, 1), 10),
        m_dir(string_t(std::getenv("HOME")) + "/experiments/databases/mnist")
{
}

void mnist_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "dir", m_dir, "folds", m_folds);
        reconfig(make_dims(1, 28, 28), make_dims(10, 1, 1), m_folds);
}

void mnist_task_t::to_json(json_t& json) const
{
        nano::to_json(json, "dir", m_dir, "folds", m_folds);
}

bool mnist_task_t::populate()
{
        const auto test_ifile = m_dir + "/t10k-images-idx3-ubyte.gz";
        const auto test_gfile = m_dir + "/t10k-labels-idx1-ubyte.gz";

        const auto train_ifile = m_dir + "/train-images-idx3-ubyte.gz";
        const auto train_gfile = m_dir + "/train-labels-idx1-ubyte.gz";

        return  load_binary(train_ifile, train_gfile, protocol::train, 60000) &&
                load_binary(test_ifile, test_gfile, protocol::test, 10000);
}

bool mnist_task_t::load_binary(const string_t& ifile, const string_t& gfile, const protocol p, const size_t count)
{
        const auto chunk_begin = n_chunks();
        const auto sample_begin = size();

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

                std::vector<tensor_size_t> ilabels;
                while (stream.read(label, 1) == 1)
                {
                        const auto ilabel = static_cast<tensor_size_t>(label[0]);
                        if (ilabel < 0 || ilabel >= nano::size(odims()))
                        {
                                log_error() << "MNIST: invalid label!";
                                return false;
                        }
                        ilabels.push_back(ilabel);
                }

                if (ilabels.size() != count)
                {
                        log_error() << "MNIST: invalid number of labels!";
                        return false;
                }

                for (size_t f = 0; f < m_folds; ++ f)
                {
                        const auto protocols = (p == protocol::train) ?
                                split2(count, protocol::train, 80, protocol::valid) :
                                std::vector<protocol>(count, protocol::test);

                        for (size_t i = 0; i < ilabels.size(); ++ i)
                        {
                                const auto ilabel = ilabels[i];
                                const auto ichunk = chunk_begin + i;
                                const auto fold = fold_t{f, protocols[i]};
                                add_sample(fold, ichunk, class_target(ilabel, nano::size(odims())), tlabels[ilabel]);
                        }
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
        log_info() << "MNIST: loaded " << (n_chunks() - chunk_begin) << " samples.";
        return  n_chunks() == chunk_begin + count &&
                size() == sample_begin + count * m_folds;
}
