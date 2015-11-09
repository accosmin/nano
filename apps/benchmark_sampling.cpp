#include "cortex/cortex.h"
#include "thread/thread.h"
#include "thread/loopi.hpp"
#include "cortex/sampler.h"
#include "cortex/util/table.h"
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
        po_desc.add_options()("help,h", "benchmark sampling");
        po_desc.add_options()("samples,s",
                boost::program_options::value<size_t>()->default_value(8000),
                "number of samples to use [256, 100000]");

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

        const size_t cmd_samples = math::clamp(po_vm["samples"].as<size_t>(), 256, 100 * 1000);

        const size_t cmd_rows = 28;
        const size_t cmd_cols = 28;
        const color_mode cmd_color = color_mode::luma;

        const size_t cmd_min_samples = cmd_samples / 8;
        const size_t cmd_max_samples = cmd_min_samples * 8;

        const size_t cmd_min_nthreads = 1;
        const size_t cmd_max_nthreads = thread::n_threads();

        // create synthetic task
        charset_task_t task(charset::numeric, cmd_rows, cmd_cols, cmd_color, cmd_samples);
        task.load("");

        tensors_t inputs(cmd_max_samples);
        vectors_t targets(cmd_max_samples);

        // construct tables to compare sampling
        table_t table("sampling\\threads");

        for (size_t nthreads = cmd_min_nthreads; nthreads <= cmd_max_nthreads; ++ nthreads)
        {
                table.header() << (text::to_string(nthreads) + "xCPU [mu]");
        }

        // evaluate sampling
        for (size_t is = cmd_min_samples; is <= cmd_max_samples; is *= 2)
        {
                const string_t cmd_name = "sample size " + text::to_string(is);

                table_row_t& row = table.append(cmd_name);

                log_info() << "<<< running test [" << cmd_name << "] ...";

                // select random samples
                sampler_t sampler(task.samples());
                sampler.push(annotation::annotated);
                sampler.push(is);

                const samples_t samples = sampler.get();

                // process the samples
                for (size_t nthreads = cmd_min_nthreads; nthreads <= cmd_max_nthreads; ++ nthreads)
                {
                        thread::pool_t pool(nthreads);

                        const auto micros = cortex::measure_robustly_usec([&]
                        {
                                thread::loopi(samples.size(), pool, [&] (size_t i)
                                {
                                        const sample_t& sample = samples[i];
                                        const image_t& image = task.image(sample.m_index);

                                        inputs[i] = image.to_tensor(sample.m_region);
                                        targets[i] = sample.m_target;
                                });
                        }, 1);

                        log_info() << "<<< processed [" << samples.size() << "] samples in " << micros << " us.";

                        row << micros;
                }

                log_info();
        }

        // print results
        table.print(std::cout);

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
