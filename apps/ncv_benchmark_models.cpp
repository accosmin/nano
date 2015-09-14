#include "libnanocv/class.h"
#include "libnanocv/nanocv.h"
#include "libnanocv/sampler.h"
#include "libnanocv/table.h"
#include "libnanocv/measure.hpp"
#include "libnanocv/accumulator.h"
#include "libmath/random.hpp"
#include "libnanocv/thread/thread.h"
#include "libtensor/random.hpp"
#include "libnanocv/tasks/task_charset.h"
#include <boost/program_options.hpp>

namespace
{
        using namespace ncv;

        void make_random_samples(
                size_t cmd_samples, size_t cmd_rows, size_t cmd_cols, size_t cmd_outputs, color_mode cmd_color,
                tensors_t& inputs, vectors_t& targets)
        {
                inputs.resize(cmd_samples);
                for (auto& input : inputs)
                {
                        input.resize(cmd_color == color_mode::luma ? 1 : 3, cmd_rows, cmd_cols);
                        tensor::set_random(input, random_t<scalar_t>(0.0, 1.0));
                }

                random_t<size_t> trgen(0, cmd_outputs);

                targets.resize(cmd_samples);
                for (auto& target : targets)
                {
                        target = ncv::class_target(trgen() % cmd_outputs, cmd_outputs);
                }
        }
}

int main(int argc, char *argv[])
{
        ncv::init();

        using namespace ncv;

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "benchmark models");
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

        const size_t cmd_samples = math::clamp(po_vm["samples"].as<size_t>(), 1000, 100 * 1000);
        const bool cmd_forward = po_vm.count("forward");
        const bool cmd_backward = po_vm.count("backward");

        if (!cmd_forward && !cmd_backward)
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const size_t cmd_rows = 28;
        const size_t cmd_cols = 28;
        const color_mode cmd_color = color_mode::luma;

        const size_t cmd_min_nthreads = 1;
        const size_t cmd_max_nthreads = ncv::n_threads();

        // generate synthetic task
        charset_task_t task(charset::numeric, cmd_rows, cmd_cols, cmd_color, cmd_samples);
        task.load("");

        // generate random samples
        tensors_t inputs;
        vectors_t targets;
        make_random_samples(cmd_samples, cmd_rows, cmd_cols, task.osize(), cmd_color, inputs, targets);

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

        const rloss_t loss = ncv::get_losses().get("logistic");
        assert(loss);

        // construct tables to compare models
        table_t ftable_rand("model-forward (rand)\\threads");
        table_t ftable_task("model-forward (task)\\threads");

        table_t btable_rand("model-backward (rand)\\threads");
        table_t btable_task("model-backward (task)\\threads");

        for (size_t nthreads = cmd_min_nthreads; nthreads <= cmd_max_nthreads; nthreads ++)
        {
                ftable_rand.header() << (text::to_string(nthreads) + "xCPU [ms]");
                ftable_task.header() << (text::to_string(nthreads) + "xCPU [ms]");

                btable_rand.header() << (text::to_string(nthreads) + "xCPU [ms]");
                btable_task.header() << (text::to_string(nthreads) + "xCPU [ms]");
        }

        // evaluate models
        for (size_t im = 0; im < cmd_networks.size(); im ++)
        {
                const string_t cmd_network = cmd_networks[im];
                const string_t cmd_name = cmd_names[im];

                table_row_t& frow_rand = ftable_rand.append(cmd_name + "(rand)");
                table_row_t& frow_task = ftable_task.append(cmd_name + "(task)");

                table_row_t& brow_rand = btable_rand.append(cmd_name + "(rand)");
                table_row_t& brow_task = btable_task.append(cmd_name + "(task)");

                log_info() << "<<< running network [" << cmd_network << "] ...";

                // create feed-forward network
                const rmodel_t model = ncv::get_models().get("forward-network", cmd_network);
                assert(model);
                model->resize(cmd_rows, cmd_cols, task.osize(), cmd_color, true);
                model->random_params();

                // select random samples
                sampler_t sampler(task);
                sampler.setup(sampler_t::stype::uniform, cmd_samples);
                sampler.setup(sampler_t::atype::annotated);

                const samples_t samples = sampler.get();

                // process the samples
                for (size_t nthreads = cmd_min_nthreads; nthreads <= cmd_max_nthreads; nthreads ++)
                {
                        if (cmd_forward)
                        {
                                accumulator_t ldata(*model, nthreads, "l2n-reg", criterion_t::type::value, 0.1);

                                const auto milis_rand = ncv::measure_robustly_usec([&] ()
                                {
                                        ldata.reset();
                                        ldata.update(inputs, targets, *loss);
                                }, 1) / 1000;

                                const auto milis_task = ncv::measure_robustly_usec([&] ()
                                {
                                        ldata.reset();
                                        ldata.update(task, samples, *loss);
                                }, 1) / 1000;

                                log_info() << "<<< processed [" << ldata.count()
                                           << "] forward samples in " << milis_rand << "/" << milis_task << " ms.";

                                frow_rand << milis_rand;
                                frow_task << milis_task;
                        }

                        if (cmd_backward)
                        {
                                accumulator_t gdata(*model, nthreads, "l2n-reg", criterion_t::type::vgrad, 0.1);

                                const auto milis_rand = ncv::measure_robustly_usec([&] ()
                                {
                                        gdata.reset();
                                        gdata.update(inputs, targets, *loss);
                                }, 1) / 1000;

                                const auto milis_task = ncv::measure_robustly_usec([&] ()
                                {
                                        gdata.reset();
                                        gdata.update(task, samples, *loss);
                                }, 1) / 1000;

                                log_info() << "<<< processed [" << gdata.count()
                                           << "] backward samples in " << milis_rand << "/" << milis_task << " ms.";

                                brow_rand << milis_rand;
                                brow_task << milis_task;
                        }
                }

                log_info();
        }

        // print results
        if (cmd_forward)
        {
                ftable_rand.print(std::cout);
                ftable_task.print(std::cout);
        }
        log_info();
        if (cmd_backward)
        {
                btable_rand.print(std::cout);
                btable_task.print(std::cout);
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
