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
                boost::program_options::value<size_t>()->default_value(1),
                "number of threads to use [1, 64], 0 - use all available threads");
        po_desc.add_options()("samples,s",
                boost::program_options::value<size_t>()->default_value(10000),
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

        const size_t cmd_threads = math::clamp(po_vm["threads"].as<size_t>(), 0, 64);
        const size_t cmd_samples = math::clamp(po_vm["samples"].as<size_t>(), 1000, 100 * 1000);
        const bool cmd_forward = po_vm.count("forward");
        const bool cmd_backward = po_vm.count("backward");

        if (!cmd_forward && !cmd_backward)
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        dummy_task_t task;
        task.set_rows(28);
        task.set_cols(28);
        task.set_color(color_mode::luma);
        task.set_outputs(10);
        task.set_folds(1);
        task.set_size(cmd_samples * 100);
        task.setup();

        const size_t cmd_outputs = task.n_outputs();

        const string_t lmodel0;
        const string_t lmodel1 = lmodel0 + "linear:dims=100;act-snorm;";
        const string_t lmodel2 = lmodel1 + "linear:dims=100;act-snorm;";
        const string_t lmodel3 = lmodel2 + "linear:dims=100;act-snorm;";
        const string_t lmodel4 = lmodel3 + "linear:dims=100;act-snorm;";
        const string_t lmodel5 = lmodel4 + "linear:dims=100;act-snorm;";
        
        string_t cmodel;
        cmodel = cmodel + "conv:dims=16,rows=6,cols=6,type=full;act-snorm;pool-max;";
        cmodel = cmodel + "conv:dims=32,rows=5,cols=5,type=full;act-snorm;pool-max;";
        cmodel = cmodel + "conv:dims=64,rows=4,cols=4,type=full;act-snorm;";

        string_t rmodel;
        rmodel = rmodel + "conv:dims=16,rows=6,cols=6,type=rand;act-snorm;pool-max;";
        rmodel = rmodel + "conv:dims=32,rows=5,cols=5,type=rand;act-snorm;pool-max;";
        rmodel = rmodel + "conv:dims=64,rows=4,cols=4,type=rand;act-snorm;";

        string_t mmodel;
        mmodel = mmodel + "conv:dims=16,rows=6,cols=6,type=mask;act-snorm;pool-max;";
        mmodel = mmodel + "conv:dims=32,rows=5,cols=5,type=mask;act-snorm;pool-max;";
        mmodel = mmodel + "conv:dims=64,rows=4,cols=4,type=mask;act-snorm;";

        const string_t outlayer = "linear:dims=" + text::to_string(cmd_outputs) + ";";

        strings_t cmd_networks =
        {
                lmodel0 + outlayer,
                lmodel1 + outlayer,
                lmodel2 + outlayer,
                lmodel3 + outlayer,
                lmodel4 + outlayer,
                lmodel5 + outlayer,

                cmodel + outlayer,
                rmodel + outlayer,
                mmodel + outlayer
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

                // select random samples
                samples_t samples;
                {
                        const ncv::timer_t timer;

                        sampler_t sampler(task);
                        sampler.setup(sampler_t::stype::uniform, cmd_samples).setup(sampler_t::atype::annotated);

                        samples = sampler.get();

                        log_info() << "<<< selected [" << samples.size()
                                   << "] random samples in " << timer.elapsed() << ".";
                }

                // simulate parameter loading & saving
                {
                        const ncv::timer_t timer;

                        vector_t params(model->psize());

                        const size_t tests = 1024;
                        for (size_t t = 0; t < tests; t ++)
                        {
                                model->save_params(params);
                                model->load_params(params);
                        }

                        log_info() << "<<< loaded & saved " << tests << "x " << model->psize()
                                   << " parameters in " << timer.elapsed() << ".";
                }

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
