#include "libnanocv/nanocv.h"
#include "libnanocv/tester.h"
#include "libnanocv/util/measure.hpp"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

using namespace ncv;

static string_t describe(const strings_t& ids)
{
        return text::concatenate(ids, ", ");
}

static string_t describe(const strings_t& ids, const strings_t& descriptions)
{
        string_t po_desc;
        for (size_t i = 0; i < ids.size(); i ++)
        {
                po_desc += "  " + text::resize(ids[i], 16) +
                           text::resize(descriptions[i], 32) + (i + 1 == ids.size() ? "" : "\n");
        }

        return po_desc;
}

int main(int argc, char *argv[])
{
        ncv::init();

        // prepare object string-based selection
        const strings_t task_ids = task_manager_t::instance().ids();
        const strings_t loss_ids = loss_manager_t::instance().ids();

        const strings_t model_ids = model_manager_t::instance().ids();
        const strings_t model_descriptions = model_manager_t::instance().descriptions();

        const strings_t trainer_ids = trainer_manager_t::instance().ids();
        const strings_t trainer_descriptions = trainer_manager_t::instance().descriptions();

        const strings_t criterion_ids = criterion_manager_t::instance().ids();
        const strings_t criterion_descriptions = criterion_manager_t::instance().descriptions();

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "help message");
        po_desc.add_options()("task",
                boost::program_options::value<string_t>(),
                describe(task_ids).c_str());
        po_desc.add_options()("task-dir",
                boost::program_options::value<string_t>(),
                "directory to load task data from");
        po_desc.add_options()("task-params",
                boost::program_options::value<string_t>()->default_value(""),
                "task parameters (if any)");
        po_desc.add_options()("loss",
                boost::program_options::value<string_t>(),
                describe(loss_ids).c_str());
        po_desc.add_options()("model",
                boost::program_options::value<string_t>(),
                describe(model_ids, model_descriptions).c_str());
        po_desc.add_options()("model-params",
                boost::program_options::value<string_t>()->default_value(""),
                "model parameters (if any) as specified in the chosen model's description");
        po_desc.add_options()("trainer",
                boost::program_options::value<string_t>(),
                describe(trainer_ids, trainer_descriptions).c_str());
        po_desc.add_options()("trainer-params",
                boost::program_options::value<string_t>()->default_value(""),
                "trainer parameters (if any) as specified in the chosen trainer's description");
        po_desc.add_options()("criterion",
                boost::program_options::value<string_t>(),
                describe(criterion_ids, criterion_descriptions).c_str());
        po_desc.add_options()("threads",
                boost::program_options::value<size_t>()->default_value(0),
                "number of threads to use (0 - all available)");
        po_desc.add_options()("trials",
                boost::program_options::value<size_t>(),
                "number of models to train & evaluate");
        po_desc.add_options()("output",
                boost::program_options::value<string_t>()->default_value(""),
                "filepath to save the best model to");
	
        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);
        		
        // check arguments and options
        if (	po_vm.empty() ||
                !po_vm.count("task") ||
                !po_vm.count("task-dir") ||
                !po_vm.count("loss") ||
                !po_vm.count("model") ||
                !po_vm.count("trainer") ||
                !po_vm.count("criterion") ||
                !po_vm.count("trials") ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const string_t cmd_task = po_vm["task"].as<string_t>();
        const string_t cmd_task_dir = po_vm["task-dir"].as<string_t>();
        const string_t cmd_task_params = po_vm["task-params"].as<string_t>();
        const string_t cmd_loss = po_vm["loss"].as<string_t>();
        const string_t cmd_model = po_vm["model"].as<string_t>();
        const string_t cmd_model_params = po_vm["model-params"].as<string_t>();
        const string_t cmd_trainer = po_vm["trainer"].as<string_t>();
        const string_t cmd_trainer_params = po_vm["trainer-params"].as<string_t>();
        const string_t cmd_criterion = po_vm["criterion"].as<string_t>();
        const size_t cmd_threads = po_vm["threads"].as<size_t>();
        const size_t cmd_trials = po_vm["trials"].as<size_t>();
        const string_t cmd_output = po_vm["output"].as<string_t>();

        // create task
        const rtask_t rtask = task_manager_t::instance().get(cmd_task, cmd_task_params);

        // load task data
        ncv::measure_critical_and_log(
                [&] () { return rtask->load(cmd_task_dir); },
                "task loaded",
                "failed to load task <" + cmd_task + "> from directory <" + cmd_task_dir + ">");

        // describe task
        rtask->describe();

        // create loss
        const rloss_t rloss = loss_manager_t::instance().get(cmd_loss);

        // create model
        const rmodel_t rmodel = model_manager_t::instance().get(cmd_model, cmd_model_params);

        // create trainer
        const rtrainer_t rtrainer = trainer_manager_t::instance().get(cmd_trainer, cmd_trainer_params);

        // train & test models
        std::map<scalar_t, std::tuple<rmodel_t, trainer_states_t>> models;

        stats_t<scalar_t> lstats, estats;
        for (size_t t = 0; t < cmd_trials; t ++)
        {
                for (size_t f = 0; f < rtask->fsize(); f ++)
                {
                        const fold_t train_fold = std::make_pair(f, protocol::train);
                        const fold_t test_fold = std::make_pair(f, protocol::test);

                        // train
                        trainer_result_t result;
                        ncv::measure_critical_and_log(
                                [&] ()
                                {
                                        result = rtrainer->train(*rtask, train_fold, *rloss, cmd_threads, cmd_criterion, *rmodel);
                                        return result.valid();
                                },
                                "model trained",
                                "failed to train model");

                        // test
                        scalar_t lvalue, lerror;
                        ncv::measure_once_and_log(
                                [&] () { ncv::test(*rtask, test_fold, *rloss, *rmodel, lvalue, lerror); },
                                "model tested");
                        log_info() << "<<< test error: [" << lvalue << "/" << lerror << "].";

                        lstats(lvalue);
                        estats(lerror);

                        // update the best model
                        models[lerror] = std::make_tuple(rmodel->clone(), result.optimum_states());
                }
        }        

        // performance statistics
        log_info() << ">>> performance: loss value = " << lstats.avg() << " +/- " << lstats.stdev()
                   << " in [" << lstats.min() << ", " << lstats.max() << "].";
        log_info() << ">>> performance: loss error = " << estats.avg() << " +/- " << estats.stdev()
                   << " in [" << estats.min() << ", " << estats.max() << "].";

        // save the best model & optimization history (if any trained)
        if (!models.empty() && !cmd_output.empty())
        {
                const rmodel_t& opt_model = std::get<0>(models.begin()->second);
                const trainer_states_t& opt_states = std::get<1>(models.begin()->second);
                
                ncv::measure_critical_and_log(
                        [&] () { return opt_model->save(cmd_output); },
                        "saved model",
                        "failed to save model to <" + cmd_output + ">");
                
                const string_t path = (boost::filesystem::path(cmd_output).parent_path() /
                        boost::filesystem::path(cmd_output).stem()).string() + ".state";
                
                ncv::measure_critical_and_log(
                        [&] () { return ncv::save(path, opt_states); },
                        "saved state",
                        "failed to save state to <" + path + ">");
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
