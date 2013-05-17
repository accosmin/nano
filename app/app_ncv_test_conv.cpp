#include "ncv.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "help message");
        po_desc.add_options()("rows",
                boost::program_options::value<ncv::size_t>()->default_value(40),
                "number of rows");
        po_desc.add_options()("cols",
                boost::program_options::value<ncv::size_t>()->default_value(40),
                "number of columns");
        po_desc.add_options()("samples",
                boost::program_options::value<ncv::size_t>()->default_value(100000),
                "number of training samples");

        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);
        		
        // check arguments and options
        if (	po_vm.empty() ||
                !po_vm.count("rows") ||
                !po_vm.count("cols") ||
                !po_vm.count("samples") ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const ncv::size_t cmd_rows = po_vm["rows"].as<ncv::size_t>();
        const ncv::size_t cmd_cols = po_vm["cols"].as<ncv::size_t>();
        const ncv::size_t cmd_samples = po_vm["samples"].as<ncv::size_t>();

        ncv::timer_t timer;

        // test convolution
        static const ncv::size_t filter_rows = 8;
        static const ncv::size_t filter_cols = 8;
        typedef ncv::tfixed_size_matrix<ncv::scalar_t, filter_rows, filter_cols>::matrix_t filter_t;

        filter_t filter;
        filter.setRandom();

        ncv::scalar_t sum = 0.0;
        for (ncv::size_t i = 0; i < cmd_samples; i ++)
        {
                ncv::matrix_t sample(cmd_rows, cmd_cols);
                sample.setRandom();

                if (cmd_rows >= filter_rows && cmd_cols >= filter_cols)
                {
                        const ncv::size_t rows = cmd_rows - filter_rows;
                        const ncv::size_t cols = cmd_cols - filter_cols;

                        ncv::matrix_t result(rows, cols);
                        for (ncv::size_t r = 0; r < rows; r ++)
                        {
                                for (ncv::size_t c = 0; c < cols; c ++)
                                {
                                        result(r, c) = (sample.block<filter_rows, filter_cols>(r, c).array() *
                                                       filter.array()).sum();
                                }
                        }

                        result /= filter.sum();
                        sum += result.sum();
                }
        }

        ncv::log_info() << "<<< convolved " << cmd_samples << " samples of size "
                        << cmd_rows << "x" << cmd_cols << " in " << timer.elapsed_string() << ".";
		
        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
