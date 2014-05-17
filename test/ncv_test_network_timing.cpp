#include "nanocv.h"
#include "models/forward_network.h"
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
        po_desc.add_options()("samples,s",
                boost::program_options::value<size_t>()->default_value(100000),
                "number of samples to use [1000, 100000]");
        po_desc.add_options()("forward",
                "evaluate the \'forward\' pass (output)");
        po_desc.add_options()("backward",
                "evaluate the \'backward' pass (gradient)");

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
        const size_t cmd_samples = math::clamp(po_vm["samples"].as<size_t>(), 1000, 100 * 1000);
        const bool cmd_forward = po_vm.count("forward");
        const bool cmd_backward = po_vm.count("backward");

        if (!cmd_forward && !cmd_backward)
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const color_mode cmd_color = color_mode::luma;
        const size_t cmd_rows = 28;
        const size_t cmd_cols = 28;
        const size_t cmd_outputs = 10;

        const string_t lmodel0;
        const string_t lmodel1 = lmodel0 + "linear:dims=64;snorm;";
        const string_t lmodel2 = lmodel1 + "linear:dims=64;snorm;";
        const string_t lmodel3 = lmodel2 + "linear:dims=64;snorm;";
        const string_t lmodel4 = lmodel3 + "linear:dims=64;snorm;";
        const string_t lmodel5 = lmodel4 + "linear:dims=64;snorm;";

        const string_t cmodel0;
        const string_t cmodel1 = cmodel0 + "conv:dims=16,rows=8,cols=8;snorm;";
        const string_t cmodel2 = cmodel1 + "conv:dims=16,rows=7,cols=7;snorm;";
        const string_t cmodel3 = cmodel2 + "conv:dims=16,rows=6,cols=6;snorm;";
        const string_t cmodel4 = cmodel3 + "conv:dims=16,rows=5,cols=5;snorm;";
        const string_t cmodel5 = cmodel4 + "conv:dims=16,rows=4,cols=4;snorm;";
        const string_t cmodel6 = cmodel5 + "conv:dims=16,rows=3,cols=3;snorm;";
        
        const string_t outlayer = "linear:dims=" + text::to_string(cmd_outputs) + ";softmax:type=global;";

        strings_t cmd_networks =
        {
                lmodel0 + outlayer,
                lmodel1 + outlayer,
                lmodel2 + outlayer,
                lmodel3 + outlayer,
                lmodel4 + outlayer,
                lmodel5 + outlayer
//                ,
//                cmodel1 + outlayer,
//                cmodel2 + outlayer,
//                cmodel3 + outlayer,
//                cmodel4 + outlayer,
//                cmodel5 + outlayer,
//                cmodel6 + outlayer
        };

        const rloss_t rloss = loss_manager_t::instance().get("class-ratio");
        assert(rloss);
        const loss_t& loss = *rloss;

        for (const string_t& cmd_network : cmd_networks)
        {
                log_info() << "<<< running network [" << cmd_network << "] ...";

                // create feed-forward network
                forward_network_t model(cmd_network);
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
                if (cmd_forward)
                {
                        accumulator_t ldata(model, accumulator_t::type::value, 0.1);

                        const ncv::timer_t timer;
                        ldata.update(samples, targets, loss, cmd_threads);

                        log_info() << "<<< processed [" << ldata.count() << "] forward samples in " << timer.elapsed() << ".";
                }

                if (cmd_backward)
                {
                        accumulator_t gdata(model, accumulator_t::type::vgrad, 0.1);

                        const ncv::timer_t timer;
                        gdata.update(samples, targets, loss, cmd_threads);

                        log_info() << "<<< processed [" << gdata.count() << "] backward samples in " << timer.elapsed() << ".";
                }
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
