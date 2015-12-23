#include "text/table.h"
#include "cortex/class.h"
#include "cortex/cortex.h"
#include "math/random.hpp"
#include "thread/thread.h"
#include "cortex/sampler.h"
#include "tensor/random.hpp"
#include "cortex/accumulator.h"
#include "cortex/util/measure.hpp"
#include "cortex/tasks/task_charset.h"
#include "cortex/util/measure_and_log.hpp"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        cortex::init();

        using namespace cortex;

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "benchmark models");
        po_desc.add_options()("samples,s",
                boost::program_options::value<size_t>()->default_value(10000),
                "number of samples to use [100, 100000]");
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

        const size_t cmd_samples = math::clamp(po_vm["samples"].as<size_t>(), 100, 100 * 1000);
        const bool cmd_forward = po_vm.count("forward");
        const bool cmd_backward = po_vm.count("backward");

        if (!cmd_forward && !cmd_backward)
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const tensor_size_t cmd_rows = 28;
        const tensor_size_t cmd_cols = 28;
        const color_mode cmd_color = color_mode::luma;

        const size_t cmd_min_nthreads = 1;
        const size_t cmd_max_nthreads = thread::n_threads();

        // generate synthetic task
        charset_task_t task(charset::numeric, cmd_rows, cmd_cols, cmd_color, cmd_samples);
        task.load("");

        // construct models
        const string_t lmodel0;
        const string_t lmodel1 = lmodel0 + "linear:dims=100;act-snorm;";
        const string_t lmodel2 = lmodel1 + "linear:dims=100;act-snorm;";
        const string_t lmodel3 = lmodel2 + "linear:dims=100;act-snorm;";
        const string_t lmodel4 = lmodel3 + "linear:dims=100;act-snorm;";
        const string_t lmodel5 = lmodel4 + "linear:dims=100;act-snorm;";

        string_t cmodel1;
        cmodel1 = cmodel1 + "conv:dims=16,rows=9,cols=9;act-snorm;";

        string_t cmodel2;
        cmodel2 = cmodel2 + "conv:dims=16,rows=9,cols=9;pool-max;act-snorm;";
        cmodel2 = cmodel2 + "conv:dims=32,rows=5,cols=5;act-snorm;";

        string_t cmodel3;
        cmodel3 = cmodel3 + "conv:dims=16,rows=9,cols=9;pool-max;act-snorm;";
        cmodel3 = cmodel3 + "conv:dims=32,rows=5,cols=5;pool-max;act-snorm;";
        cmodel3 = cmodel3 + "conv:dims=64,rows=3,cols=3;act-snorm;";

        const string_t outlayer = "linear:dims=" + text::to_string(task.osize()) + ";";

        strings_t cmd_networks =
        {
                lmodel0 + outlayer,
                lmodel1 + outlayer,
                lmodel2 + outlayer,
                lmodel3 + outlayer,
                lmodel4 + outlayer,
                lmodel5 + outlayer,

                cmodel1 + outlayer,
                cmodel2 + outlayer,
                cmodel3 + outlayer,
        };

        strings_t cmd_names =
        {
                "lmodel0",
                "lmodel1",
                "lmodel2",
                "lmodel3",
                "lmodel4",
                "lmodel5",

                "cmodel1",
                "cmodel2",
                "cmodel3"
        };

        const rloss_t loss = cortex::get_losses().get("logistic");
        assert(loss);

        // construct tables to compare models
        text::table_t ftable("model-forward");
        text::table_t btable("model-backward");

        for (size_t nthreads = cmd_min_nthreads; nthreads <= cmd_max_nthreads; ++ nthreads)
        {
                ftable.header() << (text::to_string(nthreads) + "xCPU [ms]");
                btable.header() << (text::to_string(nthreads) + "xCPU [ms]");
        }

        // evaluate models
        for (size_t im = 0; im < cmd_networks.size(); ++ im)
        {
                const string_t cmd_network = cmd_networks[im];
                const string_t cmd_name = cmd_names[im];

                text::table_row_t& frow = ftable.append(cmd_name + " (task)");
                text::table_row_t& brow = btable.append(cmd_name + " (task)");

                log_info() << "<<< running network [" << cmd_network << "] ...";

                // create feed-forward network
                const rmodel_t model = cortex::get_models().get("forward-network", cmd_network);
                assert(model);
                model->resize(cmd_rows, cmd_cols, task.osize(), cmd_color, true);
                model->random_params();

                // select random samples
                sampler_t sampler(task.samples());
                sampler.push(annotation::annotated);
                sampler.push(cmd_samples);

                const samples_t samples = sampler.get();
                size_t milis1_forward = 0;
                size_t milis1_backward = 0;

                // process the samples
                for (size_t nthreads = cmd_min_nthreads; nthreads <= cmd_max_nthreads; ++ nthreads)
                {
                        const auto op_store = [=] (auto& row, const size_t milis, size_t& milis1)
                        {
                                if (nthreads == cmd_min_nthreads)
                                {
                                        milis1 = milis;
                                        row << milis;
                                }
                                else
                                {
                                        const auto ratio = (1 + milis1) * 100 / (1 + milis);
                                        row << (text::to_string(milis) + " / " + text::to_string(ratio) + "%");
                                }
                        };

                        if (cmd_forward)
                        {
                                accumulator_t ldata(*model, nthreads, "l2n-reg", criterion_t::type::value, 0.1);

                                const auto milis = cortex::measure_robustly_usec([&] ()
                                {
                                        ldata.reset();
                                        ldata.update(task, samples, *loss);
                                }, 1) / 1000;

                                log_info() << "<<< processed [" << ldata.count()
                                           << "] forward samples in " << milis << " ms.";

                                op_store(frow, milis, milis1_forward);
                        }

                        if (cmd_backward)
                        {
                                accumulator_t gdata(*model, nthreads, "l2n-reg", criterion_t::type::vgrad, 0.1);

                                const auto milis = cortex::measure_robustly_usec([&] ()
                                {
                                        gdata.reset();
                                        gdata.update(task, samples, *loss);
                                }, 1) / 1000;

                                log_info() << "<<< processed [" << gdata.count()
                                           << "] backward samples in " << milis << " ms.";

                                op_store(brow, milis, milis1_backward);
                        }
                }

                log_info();
        }

        // print results
        if (cmd_forward)
        {
                ftable.print(std::cout);
        }
        log_info();
        if (cmd_backward)
        {
                btable.print(std::cout);
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
