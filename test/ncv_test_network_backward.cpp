#include "nanocv.h"
#include "models/forward_network.h"
#include "losses/loss_logistic.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();

        using namespace ncv;

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "test program");
        po_desc.add_options()("threads,t",
                boost::program_options::value<size_t>()->default_value(1),
                "number of threads to use [1, 16], 0 - use all available threads");

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

        const size_t cmd_threads = math::clamp(po_vm["threads"].as<size_t>(), 0, 16);

        const color_mode cmd_color = color_mode::luma;
        const size_t cmd_rows = 28;
        const size_t cmd_cols = 28;
        const size_t cmd_outputs = 10;
        const size_t cmd_samples = 10000;

        string_t lmodel0;
        string_t lmodel1 = lmodel0 + "linear:dims=1000;snorm;";
        string_t lmodel2 = lmodel1 + "linear:dims=100;snorm;";
        string_t lmodel3 = lmodel2 + "linear:dims=1000;snorm;";
        string_t lmodel4 = lmodel3 + "linear:dims=100;snorm;";
        string_t lmodel5 = lmodel4 + "linear:dims=1000;snorm;";

        string_t cmodel1;
        cmodel1 = cmodel1 + "conv:dims=32,rows=7,cols=7;snorm;smax-abs-pool;";
        cmodel1 = cmodel1 + "conv:dims=32,rows=4,cols=4;snorm;smax-abs-pool;";
        cmodel1 = cmodel1 + "conv:dims=32,rows=4,cols=4;snorm;";

        string_t cmodel2;
        cmodel2 = cmodel2 + "conv:dims=32,rows=7,cols=7;snorm;smax-abs-pool;";
        cmodel2 = cmodel2 + "conv:dims=32,rows=6,cols=6;snorm;smax-abs-pool;";
        cmodel2 = cmodel2 + "conv:dims=32,rows=3,cols=3;snorm;";

        string_t cmodel3;
        cmodel3 = cmodel3 + "conv:dims=32,rows=5,cols=5;snorm;smax-abs-pool;";
        cmodel3 = cmodel3 + "conv:dims=32,rows=5,cols=5;snorm;smax-abs-pool;";
        cmodel3 = cmodel3 + "conv:dims=32,rows=4,cols=4;snorm;";

        strings_t cmd_networks =
        {
//                lmodel0,
//                lmodel1,
//                lmodel2,
//                lmodel3,
//                lmodel4,
//                lmodel5,

                cmodel1,
                cmodel2,
                cmodel3
        };

        const logistic_loss_t loss;

        for (const string_t& cmd_network : cmd_networks)
        {
                log_info() << "<<< running network [" << cmd_network << "] ...";

                // create feed-forward network
                ncv::forward_network_t model(cmd_network);
                model.resize(cmd_rows, cmd_cols, cmd_outputs, cmd_color, true);

                // create random samples
                tensors_t samples(cmd_samples, tensor_t(cmd_color == color_mode::luma ? 1 : 3, cmd_rows, cmd_cols));

                random_t<scalar_t> rgen(-1.0, +1.0);
                for (tensor_t& sample : samples)
                {
                        sample.random(rgen);
                }

                // create random targets
                vectors_t targets(cmd_samples, vector_t(cmd_outputs));
                for (vector_t& target : targets)
                {
                        target.setRandom();
                }

                // process the samples
                ncv::timer_t timer;

                trainer_data_t gdata(model, trainer_data_t::type::vgrad);
                gdata.update_mt(samples, targets, loss, cmd_threads);

                log_info() << "<<< processed [" << gdata.count() << "] samples in " << timer.elapsed() << ".";
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
