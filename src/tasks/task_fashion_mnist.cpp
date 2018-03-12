#include "logger.h"
#include "io/archive.h"
#include "task_fashion_mnist.h"

using namespace nano;

static const string_t tlabels[] =
{
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
};

fashion_mnist_task_t::fashion_mnist_task_t() :
        mem_vision_task_t(tensor3d_dim_t{1, 28, 28}, tensor3d_dim_t{10, 1, 1}, 1),
        m_dir(string_t(std::getenv("HOME")) + "/experiments/databases/fashion-mnist")
{
}

void fashion_mnist_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "dir", m_dir);
}

void fashion_mnist_task_t::to_json(json_t& json) const
{
        nano::to_json(json, "dir", m_dir);
}

bool fashion_mnist_task_t::populate()
{
        const string_t test_ifile = m_dir + "/t10k-images-idx3-ubyte.gz";
        const string_t test_gfile = m_dir + "/t10k-labels-idx1-ubyte.gz";
        const size_t n_test_samples = 10000;

        const string_t train_ifile = m_dir + "/train-images-idx3-ubyte.gz";
        const string_t train_gfile = m_dir + "/train-labels-idx1-ubyte.gz";
        const size_t n_train_samples = 60000;

        return  load_binary(train_ifile, train_gfile, protocol::train, n_train_samples) &&
                load_binary(test_ifile, test_gfile, protocol::test, n_test_samples);
}

bool fashion_mnist_task_t::load_binary(const string_t& ifile, const string_t& gfile,
        const protocol p, const size_t count)
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
                log_error() << "Fashion-MNIST: " << message;
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

        log_info() << "Fashion-MNIST: loading file <" << ifile << "> ...";
        if (!nano::load_archive(ifile, iop, error_op))
        {
                log_error() << "Fashion-MNIST: failed to load file <" << ifile << ">!";
                return false;
        }

        // load ground truth
        const auto gop = [&] (const string_t&, istream_t& stream)
        {
                if (stream.read(buffer.data(), 8) != 8)
                {
                        return false;
                }

                while (stream.read(label, 1) == 1)
                {
                        const auto ilabel = static_cast<tensor_size_t>(label[0]);
                        if (ilabel < 0 || ilabel >= nano::size(odims()))
                        {
                                log_error() << "Fashion-MNIST: invalid label!";
                                return false;
                        }

                        const auto fold = make_fold(0, p);
                        add_sample(fold, iindex, class_target(ilabel, nano::size(odims())), tlabels[ilabel]);

                        ++ gcount;
                        ++ iindex;
                }

                return true;
        };

        log_info() << "Fashion-MNIST: loading file <" << gfile << "> ...";
        if (!nano::load_archive(gfile, gop, error_op))
        {
                log_error() << "Fashion-MNIST: failed to load file <" << gfile << ">!";
                return false;
        }

        // OK
        log_info() << "Fashion-MNIST: loaded " << icount << "/" << gcount << " samples.";
        return (count == gcount) && (count == icount);
}
