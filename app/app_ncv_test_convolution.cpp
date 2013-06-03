#include "ncv.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "help message");
        po_desc.add_options()("samples",
                boost::program_options::value<ncv::size_t>()->default_value(10000),
                "number of samples to evaluate [1K, 100K]");
        po_desc.add_options()("size",
                boost::program_options::value<ncv::size_t>()->default_value(32),
                "sample size in pixels [16, 64]");
        po_desc.add_options()("filters",
                boost::program_options::value<ncv::size_t>()->default_value(8),
                "number of filters [1, 64]");

        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);

        // check arguments and options
        if (	po_vm.empty() ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const ncv::size_t cmd_samples = ncv::math::clamp(po_vm["samples"].as<ncv::size_t>(), 1024, 100 * 1024);
        const ncv::size_t cmd_size = ncv::math::clamp(po_vm["size"].as<ncv::size_t>(), 16, 64);
        const ncv::size_t cmd_filters = ncv::math::clamp(po_vm["filters"].as<ncv::size_t>(), 1, 64);

        // generate random samples
        ncv::random_t<ncv::scalar_t> rgen(-1.0, +1.0);

        ncv::matrices_t samples(cmd_samples, ncv::matrix_t(cmd_size, cmd_size));
        for (ncv::size_t s = 0; s < cmd_samples; s ++)
        {
                ncv::matrix_t& sample = samples[s];
                rgen(sample.data(), sample.data() + sample.size());
        }

        // generate random filters
        ncv::matrices_t filters(cmd_filters, ncv::matrix_t(cmd_size, cmd_size));
        for (ncv::size_t f = 0; f < cmd_filters; f ++)
        {
                ncv::matrix_t& filter = filters[f];
                rgen(filter.data(), filter.data() + filter.size());
        }

        // evaluate convolution
        ncv::timer_t timer;

        ncv::scalar_t sum_total = 0.0;
        for (ncv::size_t s = 0; s < cmd_samples; s ++)
        {
                const ncv::matrix_t& sample = samples[s];

                for (ncv::size_t f = 0; f < cmd_filters; f ++)
                {
                        const ncv::matrix_t& filter = filters[f];

                        const ncv::scalar_t sum_conv = filter.cwiseProduct(sample).sum();
                        sum_total += sum_conv;
                }

//                ncv::scalar_t sum = 0.0;
//                for (ncv::size_t r = 0; r < cmd_size; r ++)
//                {
//                        for (ncv::size_t c = 0; c < cmd_size; c ++)
//                        {
//                                sum += filter(r, c) * sample(r, c);
//                        }
//                }

//                std::cout << "sum = " << sum << ", sum_conv = " << sum_conv << std::endl;
        }

        ncv::log_info() << "(" << sum_total << ") convolved " << cmd_samples << " ["
                   << cmd_size << "x" << cmd_size << "] samples in " << timer.elapsed();

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
