#include "ncv_task_mnist.h"
#include "ncv_random.h"
#include "ncv_loss.h"
#include "ncv_color.h"
#include <fstream>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        mnist_task::mnist_task()
        {
                for (size_t i = 0; i < 10; i ++)
                {
                        m_labels.push_back("digit-" + text::to_string(i));
                }
        }

        //-------------------------------------------------------------------------------------------------

        mnist_task::~mnist_task()
        {
        }

        //-------------------------------------------------------------------------------------------------

        bool mnist_task::load(
                const string_t& dir,
                size_t ram_gb,
                samples_t& train_samples,
                samples_t& valid_samples,
                samples_t& test_samples)
        {
                if (ram_gb < 1)
                {
                        return false;
                }

                const string_t test_ifile = dir + "/t10k-images-idx3-ubyte";
                const string_t test_lfile = dir + "/t10k-labels-idx1-ubyte";
                const size_t test_n_samples = 10000;

                const string_t train_ifile = dir + "/train-images-idx3-ubyte";
                const string_t train_lfile = dir + "/train-labels-idx1-ubyte";
                const size_t train_n_samples = 60000;

                // clear
                train_samples.clear();
                valid_samples.clear();
                test_samples.clear();

                // (60000 + 10000) *
                //      (28 * 28 * sizeof(scalar_t) +
                //            10 * sizeof(scalar_t) +
                //             1 * sizeof(scalar_t)) < 1GB

                // TODO: set costs!
                return  load(train_ifile, train_lfile, train_samples, valid_samples, 0.84) &&
                        load(test_ifile, test_lfile, test_samples, test_samples, 2.0) &&

                        train_samples.size() + valid_samples.size() == train_n_samples &&
                        test_samples.size() == test_n_samples;
        }

        //-------------------------------------------------------------------------------------------------
        
        bool mnist_task::load(const string_t& ifile, const string_t& gfile,
                samples_t& samples1, samples_t& samples2, scalar_t prob)
        {
                static const index_t n_inputs = n_rows() * n_cols();

                char buffer[2048];
                char label[2];

                random<scalar_t> rgen(0.0, 1.0);
                
                // image and label data streams
                std::ifstream fimage(ifile.c_str(), std::ios::in | std::ios::binary);
                std::ifstream flabel(gfile.c_str(), std::ios::in | std::ios::binary);
                
                if (!fimage.is_open() || !flabel.is_open())
                {
                        return false;
                }                

                // read headers
                fimage.read(buffer, 16);
                flabel.read(buffer, 8);
                
                // load images
                while ( flabel.read(label, 1) &&
                        fimage.read(buffer, n_inputs))
                {
                        // setup label
                        const index_t ilabel = static_cast<index_t>(label[0]);
                        if (ilabel >= n_labels())
                        {
                                continue;
                        }

                        // setup sample
                        sample s;
                        s.m_label = m_labels[ilabel];
                        s.m_weight = 1.0;
                        s.m_input.resize(n_rows(), n_cols());
                        s.m_target = ncv::class_target(ilabel, n_labels());

                        for (index_t y = 0, i = 0; y < n_rows(); y ++)
                        {
                                for (index_t x = 0; x < n_cols(); x ++, i ++)
                                {
                                        const rgba_t rgba = color::make_rgba(buffer[i], buffer[i], buffer[i]);
                                        const cielab_t cielab = color::make_cielab(rgba);
                                        s.m_input(y, x) = cielab(0);
                                }
                        }

                        // OK, add sample
                        (rgen() < prob ? samples1 : samples2).push_back(s);
                }
                
                // OK
                return true;
        }

        //-------------------------------------------------------------------------------------------------
}
