#include "nanocv.h"
#include "models/forward_network.h"
#include "task.h"
#include <boost/program_options.hpp>

namespace ncv 
{
        class dummy_tast_t : public ncv::task_t
        {
        public:
                
                NANOCV_MAKE_CLONABLE(dummy_tast_t)
                
                // constructor
                dummy_tast_t(const string_t& = string_t())
                        :       task_t("test task")
                {        
                }
                
                // create samples
                void resize(size_t samples)
                {
                        m_images.clear();
                        m_samples.clear();
                        
                        for (size_t i = 0; i < samples; i ++)
                        {                
                                sample_t sample(m_images.size(), sample_region(0, 0));
                                sample.m_label = "label";
                                sample.m_target = ncv::class_target(i % n_outputs(), n_outputs());
                                sample.m_fold = { 0, protocol::train };
                                m_samples.push_back(sample);

                                image_t image(n_rows(), n_cols(), color());
                                m_images.push_back(image);
                        }                        
                }                        
                
                // load images from the given directory
                virtual bool load(const string_t&) { return true; }
                
                // access functions
                virtual size_t n_rows() const { return 28; }
                virtual size_t n_cols() const { return 28; }
                virtual size_t n_outputs() const { return 10; }
                virtual size_t n_folds() const { return 1; }
                virtual color_mode color() const { return color_mode::luma; }
        };
}

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
        
        dummy_tast_t task;
        task.resize(cmd_samples * 100);

        const size_t cmd_outputs = task.n_outputs();

        const string_t lmodel0;
        const string_t lmodel1 = lmodel0 + "linear:dims=128;act-snorm;";
        const string_t lmodel2 = lmodel1 + "linear:dims=64;act-snorm;";
        const string_t lmodel3 = lmodel2 + "linear:dims=32;act-snorm;";
        const string_t lmodel4 = lmodel3 + "linear:dims=16;act-snorm;";
        const string_t lmodel5 = lmodel4 + "linear:dims=8;act-snorm;";
        
        string_t cmodel;
        cmodel = cmodel + "conv:dims=16,rows=7,cols=7;act-snorm;pool-max;";
        cmodel = cmodel + "conv:dims=32,rows=5,cols=5;act-snorm;pool-max;";
        cmodel = cmodel + "conv:dims=64,rows=3,cols=3;act-snorm;";
        
        const string_t outlayer = "linear:dims=" + text::to_string(cmd_outputs) + ";softmax:type=global;";

        strings_t cmd_networks =
        {
                lmodel0 + outlayer,
                lmodel1 + outlayer,
                lmodel2 + outlayer,
                lmodel3 + outlayer,
                lmodel4 + outlayer,
                lmodel5 + outlayer,
                cmodel + outlayer
        };

        const rloss_t rloss = loss_manager_t::instance().get("class-ratio");
        assert(rloss);
        const loss_t& loss = *rloss;

        for (const string_t& cmd_network : cmd_networks)
        {
                log_info() << "<<< running network [" << cmd_network << "] ...";

                // create feed-forward network
                forward_network_t model(cmd_network);
                model.resize(task, true);

                // select random samples
                samples_t samples;
                {
                        const ncv::timer_t timer;

                        sampler_t sampler(task);
                        sampler.setup(sampler_t::stype::uniform, cmd_samples).setup(sampler_t::atype::annotated);

                        samples = sampler.get();

                        log_info() << "<<< selected [" << samples.size() << "] random samples in " << timer.elapsed() << ".";
                }

                // process the samples
                if (cmd_forward)
                {
                        accumulator_t ldata(model, cmd_threads, "l2-reg", criterion_t::type::value, 0.1);

                        const ncv::timer_t timer;
                        ldata.update(task, samples, loss);

                        log_info() << "<<< processed [" << ldata.count() << "] forward samples in " << timer.elapsed() << ".";
                }

                if (cmd_backward)
                {
                        accumulator_t gdata(model, cmd_threads, "l2-reg", criterion_t::type::vgrad, 0.1);

                        const ncv::timer_t timer;
                        gdata.update(task, samples, loss);

                        log_info() << "<<< processed [" << gdata.count() << "] backward samples in " << timer.elapsed() << ".";
                }
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
