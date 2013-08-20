#include "ncv.h"
#include "core/clamp.hpp"
#include "core/timer.h"
#include "core/logger.h"
#include "core/tensor3d.h"
#include "models/forward_network.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();

        using namespace ncv;

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "help message");
        po_desc.add_options()("inputs,i",
                boost::program_options::value<string_t>()->default_value("rgba"),
                "number of inputs [luma, rgba]");
        po_desc.add_options()("rows,r",
                boost::program_options::value<size_t>()->default_value(32),
                "number of input rows [16, 64]");
        po_desc.add_options()("cols,c",
                boost::program_options::value<size_t>()->default_value(32),
                "number of input columns [16, 64]");
        po_desc.add_options()("outputs,o",
                boost::program_options::value<size_t>()->default_value(10),
                "number of input rows [1, 100]");
        po_desc.add_options()("network,n",
                boost::program_options::value<string_t>()->default_value(""),
                "network parameters");
        po_desc.add_options()("samples,s",
                boost::program_options::value<size_t>()->default_value(10000),
                "number of random samples [1K, 1M]");

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

        const color_mode cmd_color = text::from_string<color_mode>(po_vm["inputs"].as<string_t>());
        const size_t cmd_rows = math::clamp(po_vm["rows"].as<size_t>(), 16, 64);
        const size_t cmd_cols = math::clamp(po_vm["cols"].as<size_t>(), 16, 64);
        const size_t cmd_outputs = math::clamp(po_vm["outputs"].as<size_t>(), 1, 100);
        const string_t cmd_network = po_vm["network"].as<string_t>();
        const size_t cmd_samples = math::clamp(po_vm["samples"].as<size_t>(), 1000, 1000 * 1000);

        // create feed-forward network
        ncv::forward_network_t network(cmd_network);
        network.resize(cmd_rows, cmd_cols, cmd_outputs, cmd_color);

        // create random samples
        tensor3ds_t samples(cmd_samples, tensor3d_t(cmd_color == color_mode::luma ? 1 : 3, cmd_rows, cmd_cols));
        for (tensor3d_t& sample : samples)
        {
                sample.random(-1.0, +1.0);
        }

        // process the samples
        ncv::timer_t timer;

        size_t count = 0;
        for (tensor3d_t& sample : samples)
        {
                const vector_t output = network.value(sample);
                count += output.size();
        }

        log_info() << "<<< processed [" << (count / cmd_outputs) << "] samples in " << timer.elapsed() << ".";

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
