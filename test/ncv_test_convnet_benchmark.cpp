#include "nanocv.h"
#include "tasks/task_dummy.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();

        using namespace ncv;

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "test program");
        po_desc.add_options()("threads,t",
                boost::program_options::value<size_t>()->default_value(0),
                "number of threads to use [1, 64], 0 - use all available threads");
        po_desc.add_options()("samples,s",
                boost::program_options::value<size_t>()->default_value(128),
                "number of samples to use [16, 2048]");

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

        const size_t cmd_threads = math::clamp(po_vm["threads"].as<size_t>(), 0, 64);
        const size_t cmd_samples = math::clamp(po_vm["samples"].as<size_t>(), 16, 2048);
        const bool cmd_forward = true;
        const bool cmd_backward = true;

        if (!cmd_forward && !cmd_backward)
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        dummy_task_t task;
        task.set_rows(128);
        task.set_cols(128);
        task.set_color(color_mode::rgba);
        task.set_outputs(10);
        task.set_folds(1);
        task.set_size(cmd_samples);
        task.setup();

        const size_t cmd_outputs = task.n_outputs();

        string_t cmodel;
        cmodel = cmodel + "conv:dims=96,rows=11,cols=11,type=full;act-snorm;pool-max;";
        cmodel = cmodel + "conv:dims=64,rows=9,cols=9,type=full;act-snorm;pool-max;";
        cmodel = cmodel + "conv:dims=128,rows=9,cols=9,type=full;act-snorm;pool-max;";
        cmodel = cmodel + "conv:dims=128,rows=7,cols=7,type=full;act-snorm;pool-max;";
        cmodel = cmodel + "conv:dims=384,rows=2,cols=2,type=full;act-snorm;";

        const string_t outlayer = "linear:dims=" + text::to_string(cmd_outputs) + ";";

        strings_t cmd_networks =
        {
                cmodel + outlayer
        };

        const rloss_t loss = loss_manager_t::instance().get("logistic");
        assert(loss);

        for (const string_t& cmd_network : cmd_networks)
        {
                log_info() << "<<< running network [" << cmd_network << "] ...";

                // create feed-forward network
                const rmodel_t model = model_manager_t::instance().get("forward-network", cmd_network);
                assert(model);
                model->resize(task, true);

                const samples_t& samples = task.samples();

                // process the samples
                if (cmd_forward)
                {
                        accumulator_t ldata(*model, cmd_threads, "l2n-reg", criterion_t::type::value, 0.1);

                        const ncv::timer_t timer;
                        ldata.update(task, samples, *loss);

                        log_info() << "<<< processed [" << ldata.count() << "] forward samples in " << timer.elapsed() << ".";
                }

                if (cmd_backward)
                {
                        accumulator_t gdata(*model, cmd_threads, "l2n-reg", criterion_t::type::vgrad, 0.1);

                        const ncv::timer_t timer;
                        gdata.update(task, samples, *loss);

                        log_info() << "<<< processed [" << gdata.count() << "] backward samples in " << timer.elapsed() << ".";
                }

                log_info();
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
