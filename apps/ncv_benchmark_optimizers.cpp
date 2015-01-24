#include "nanocv.h"
#include "criterion.h"
#include "accumulator.h"
#include "tasks/task_syn_dots.h"
#include "util/timer.h"
#include "util/logger.h"
#include "util/random.hpp"
#include "util/log_search.hpp"
#include "util/thread_loop.hpp"
#include "optimize/batch_gd.hpp"
#include "optimize/batch_cgd.hpp"
#include "optimize/batch_lbfgs.hpp"
#include "optimize/stoch_ag.hpp"
#include "optimize/stoch_sg.hpp"
#include "optimize/stoch_sga.hpp"
#include "optimize/stoch_sia.hpp"
#include "optimize/stoch_adagrad.hpp"
#include "optimize/stoch_adadelta.hpp"
#include <map>

using namespace ncv;

const size_t cmd_trials = 256;

// optimization statistics for a particular algorithm
struct opt_info_t
{
        opt_info_t(const string_t& name = string_t())
                :       m_name(name),
                        m_miliseconds(0),
                        m_score(0.0),
                        m_count(0)
        {
        }
        
        void update(scalar_t loss, scalar_t best_loss, size_t miliseconds)
        {
                assert(loss > 0.0);
                assert(best_loss > 0.0);
                assert(loss >= best_loss);

                m_miliseconds += miliseconds;
                m_score += std::log(best_loss / loss);
                m_count ++;
        }
        
        string_t        m_name;                 ///< algorithm name
        size_t          m_miliseconds;          ///< total amount of time (miliseconds)
        scalar_t        m_score;                ///< optimization score (relative to the best optimizer)
        size_t          m_count;                ///<
};

bool operator<(const opt_info_t& info1, const opt_info_t& info2)
{
        return info1.m_score >= info2.m_score;
}

typedef std::vector<opt_info_t>         opt_infos_t;

// search for an optimization algorithm and create a new one if not found
opt_info_t& giveit(opt_infos_t& infos, const string_t& name)
{
        for (opt_info_t& info : infos)
        {
                if (info.m_name == name)
                {
                        return info;
                }
        }

        infos.push_back(opt_info_t(name));
        return *infos.rbegin();
}

// display the formatted optimization statistics for all optimization algorithms
void print_all(const string_t& name, const opt_infos_t& infos)
{
        static const char delim = '$';
        static const size_t col_size = 24;
        static const string_t delim_str = string_t(1, delim);
        static const string_t delim_line(3 * col_size, delim);
        
        std::cout << delim_line << std::endl;
        std::cout << text::resize(" " + name + " ", delim_line.size(), align::center, delim) << std::endl;
        std::cout << delim_line << std::endl;
        std::cout << text::resize(delim_str + " " + "[algorithm]", col_size);
        std::cout << text::resize("[cumulated score]", col_size);
        std::cout << text::resize("[total time (ms)]", col_size) << std::endl;

        for (const opt_info_t& info : infos)
        {
                std::cout << text::resize(delim_str + " " + info.m_name, col_size);
                std::cout << text::resize(text::to_string(info.m_score), col_size);
                std::cout << text::resize(text::to_string(info.m_miliseconds), col_size);
                std::cout << std::endl;
        }
        std::cout << delim_line << std::endl;
}

template <typename toptimizer>
std::tuple<scalar_t, string_t, size_t> batch(
        const task_t& task, const model_t& model, const loss_t& loss, const string_t& criterion,
        toptimizer& optimizer, const string_t& header, const string_t& name)
{
        samples_t samples = task.samples();

        accumulator_t ldata(model, 1, criterion, criterion_t::type::value, 0.1);
        accumulator_t gdata(model, 1, criterion, criterion_t::type::vgrad, 0.1);

        vector_t x0;
        model.save_params(x0);

        // construct the optimization problem
        const ncv::timer_t timer;

        auto fn_size = [&] ()
        {
                return ldata.psize();
        };

        auto fn_fval = [&] (const vector_t& x)
        {
                ldata.reset(x);
                ldata.update(task, samples, loss);

                return ldata.value();
        };

        auto fn_grad = [&] (const vector_t& x, vector_t& gx)
        {
                gdata.reset(x);
                gdata.update(task, samples, loss);

                gx = gdata.vgrad();
                return gdata.value();
        };

        // assembly optimization problem & optimize the model
        const opt_problem_t problem(fn_size, fn_fval, fn_grad);

        optimizer.set_wlog(nullptr);
        optimizer.set_elog(nullptr);
        optimizer.set_ulog(nullptr);

        const opt_state_t state = optimizer(problem, x0);

//        log_info() << name << ": value = " << state.f << ", done in " << timer.elapsed() << ".";

        // OK
        return std::make_tuple(state.f, name, timer.miliseconds());
}

template <typename toptimizer>
std::tuple<scalar_t, string_t, size_t> stoch(
        const task_t& task, const model_t& model, const loss_t& loss, const string_t& criterion,
        toptimizer& optimizer, const string_t& header, const string_t& name)
{
        const ncv::timer_t timer;

        // tune the learning rate
        const auto op_tune_alpha0 = [&] (scalar_t alpha0)
        {
                samples_t samples = task.samples();

                accumulator_t ldata(model, 1, criterion, criterion_t::type::value, 0.1);
                accumulator_t gdata(model, 1, criterion, criterion_t::type::vgrad, 0.1);

                vector_t x0;
                model.save_params(x0);

                // construct the optimization problem (NB: one random sample at the time)
                size_t index = 0;

                auto fn_size = [&] ()
                {
                        return ldata.psize();
                };

                auto fn_fval = [&] (const vector_t& x)
                {
                        ldata.reset(x);
                        ldata.update(task, samples[(index ++) % samples.size()], loss);

                        return ldata.value();
                };

                auto fn_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        gdata.reset(x);
                        gdata.update(task, samples[(index ++) % samples.size()], loss);

                        gx = gdata.vgrad();
                        return gdata.value();
                };

                const opt_opulog_t fn_ulog = [&] (const opt_state_t&)
                {
                        // shuffle randomly the training samples after each epoch
                        random_t<size_t> xrng(0, samples.size());
                        random_index_t<size_t> xrnd(xrng);

                        std::random_shuffle(samples.begin(), samples.end(), xrnd);
                };

                // assembly optimization problem & optimize the model
                const opt_problem_t problem(fn_size, fn_fval, fn_grad);

                toptimizer optimizer_copy(optimizer);
                optimizer_copy.set_ulog(fn_ulog);
                optimizer_copy.set_alpha0(alpha0);

                opt_state_t state = optimizer_copy(problem, x0);

                ldata.reset(state.x);
                ldata.update(task, samples, loss);
                state.f = ldata.value();        // need to compute the cumulative loss for stochastic optimizers

                return state;
        };

        const opt_state_t state = ncv::log10_min_search(op_tune_alpha0, -6.0, 0.0, 0.2, 4).first;

//        log_info() << header << "[" + name << "]: value = " << state.f << ", done in " << timer.elapsed() << ".";

        // OK
        return std::make_tuple(state.f, name, timer.miliseconds());
}

void test_optimize(
        const task_t& task, const model_t& imodel, const loss_t& loss, const string_t& criterion,
        const string_t& config_name)
{
        const size_t cmd_iterations = 128;
        const scalar_t cmd_epsilon = 1e-6;

        const size_t cmd_epochs = cmd_iterations;
        const size_t cmd_epoch_size = task.samples().size();

        // create batch optimizers
        auto batch_gd = optimize::batch_gd<opt_problem_t>(cmd_iterations, cmd_epsilon);

        auto batch_cgd_n = optimize::batch_cgd_n<opt_problem_t>(cmd_iterations, cmd_epsilon);
        auto batch_cgd_cd = optimize::batch_cgd_cd<opt_problem_t>(cmd_iterations, cmd_epsilon);
        auto batch_cgd_dy = optimize::batch_cgd_dy<opt_problem_t>(cmd_iterations, cmd_epsilon);
        auto batch_cgd_fr = optimize::batch_cgd_fr<opt_problem_t>(cmd_iterations, cmd_epsilon);
        auto batch_cgd_hs = optimize::batch_cgd_hs<opt_problem_t>(cmd_iterations, cmd_epsilon);
        auto batch_cgd_ls = optimize::batch_cgd_ls<opt_problem_t>(cmd_iterations, cmd_epsilon);
        auto batch_cgd_pr = optimize::batch_cgd_pr<opt_problem_t>(cmd_iterations, cmd_epsilon);

        auto batch_lbfgs6 = optimize::batch_lbfgs<opt_problem_t>(cmd_iterations, cmd_epsilon, 6);
        auto batch_lbfgs8 = optimize::batch_lbfgs<opt_problem_t>(cmd_iterations, cmd_epsilon, 8);
        auto batch_lbfgs10 = optimize::batch_lbfgs<opt_problem_t>(cmd_iterations, cmd_epsilon, 10);
        auto batch_lbfgs12 = optimize::batch_lbfgs<opt_problem_t>(cmd_iterations, cmd_epsilon, 12);
        auto batch_lbfgs16 = optimize::batch_lbfgs<opt_problem_t>(cmd_iterations, cmd_epsilon, 16);
        auto batch_lbfgs20 = optimize::batch_lbfgs<opt_problem_t>(cmd_iterations, cmd_epsilon, 20);

        // create stochastic optimizers
        auto stoch_ag = optimize::stoch_ag<opt_problem_t>(cmd_epochs, cmd_epoch_size, 0.1, 1.00);

        auto stoch_adagrad = optimize::stoch_adagrad<opt_problem_t>(cmd_epochs, cmd_epoch_size, 0.1, 1.00);
        auto stoch_adadelta = optimize::stoch_adadelta<opt_problem_t>(cmd_epochs, cmd_epoch_size, 0.1, 1.00);

        auto stoch_sg050 = optimize::stoch_sg<opt_problem_t>(cmd_epochs, cmd_epoch_size, 0.1, 0.50);
        auto stoch_sg075 = optimize::stoch_sg<opt_problem_t>(cmd_epochs, cmd_epoch_size, 0.1, 0.75);
        auto stoch_sg100 = optimize::stoch_sg<opt_problem_t>(cmd_epochs, cmd_epoch_size, 0.1, 1.00);

        auto stoch_sga050 = optimize::stoch_sga<opt_problem_t>(cmd_epochs, cmd_epoch_size, 0.1, 0.50);
        auto stoch_sga075 = optimize::stoch_sga<opt_problem_t>(cmd_epochs, cmd_epoch_size, 0.1, 0.75);
        auto stoch_sga100 = optimize::stoch_sga<opt_problem_t>(cmd_epochs, cmd_epoch_size, 0.1, 1.00);

        auto stoch_sia050 = optimize::stoch_sia<opt_problem_t>(cmd_epochs, cmd_epoch_size, 0.1, 0.50);
        auto stoch_sia075 = optimize::stoch_sia<opt_problem_t>(cmd_epochs, cmd_epoch_size, 0.1, 0.75);
        auto stoch_sia100 = optimize::stoch_sia<opt_problem_t>(cmd_epochs, cmd_epoch_size, 0.1, 1.00);

        opt_infos_t infos;

        thread_pool_t::mutex_t mutex;

        // optimize starting from various random initial parameters
        ncv::thread_loopi(cmd_trials, [&] (size_t cmd_trial)
        {
                // random initialization
                const rmodel_t rmodel = imodel.clone();
                model_t& model = *rmodel;
                model.random_params();

                const string_t header = "[" + text::to_string(cmd_trial) + "/" + text::to_string(cmd_trials) + "] ";

                std::vector<std::tuple<scalar_t, string_t, size_t>> results;

                // batch optimizers
                results.push_back(batch(task, model, loss, criterion, batch_gd, header, "batch-GD"));

                results.push_back(batch(task, model, loss, criterion, batch_cgd_n, header, "batch-CGD-N"));
                results.push_back(batch(task, model, loss, criterion, batch_cgd_cd, header, "batch-CGD-CD"));
                results.push_back(batch(task, model, loss, criterion, batch_cgd_dy, header, "batch-CGD-DY"));
                results.push_back(batch(task, model, loss, criterion, batch_cgd_fr, header, "batch-CGD-FR"));
                results.push_back(batch(task, model, loss, criterion, batch_cgd_hs, header, "batch-CGD-HS"));
                results.push_back(batch(task, model, loss, criterion, batch_cgd_ls, header, "batch-CGD-LS"));
                results.push_back(batch(task, model, loss, criterion, batch_cgd_pr, header, "batch-CGD-PR"));

                results.push_back(batch(task, model, loss, criterion, batch_lbfgs6, header, "batch-LBFGS-6"));
                results.push_back(batch(task, model, loss, criterion, batch_lbfgs8, header, "batch-LBFGS-8"));
                results.push_back(batch(task, model, loss, criterion, batch_lbfgs10, header, "batch-LBFGS-10"));
                results.push_back(batch(task, model, loss, criterion, batch_lbfgs12, header, "batch-LBFGS-12"));
                results.push_back(batch(task, model, loss, criterion, batch_lbfgs16, header, "batch-LBFGS-16"));
                results.push_back(batch(task, model, loss, criterion, batch_lbfgs20, header, "batch-LBFGS-20"));

                // stochastic optimizers
                results.push_back(stoch(task, model, loss, criterion, stoch_ag, header, "stoch-AG"));

                results.push_back(stoch(task, model, loss, criterion, stoch_adagrad, header, "stoch-ADAGRAD"));
                results.push_back(stoch(task, model, loss, criterion, stoch_adadelta, header, "stoch-ADADELTA"));

                results.push_back(stoch(task, model, loss, criterion, stoch_sg050, header, "stoch-SG-0.50"));
                results.push_back(stoch(task, model, loss, criterion, stoch_sg075, header, "stoch-SG-0.75"));
                results.push_back(stoch(task, model, loss, criterion, stoch_sg100, header, "stoch-SG-1.00"));

                results.push_back(stoch(task, model, loss, criterion, stoch_sga050, header, "stoch-SGA-0.50"));
                results.push_back(stoch(task, model, loss, criterion, stoch_sga075, header, "stoch-SGA-0.75"));
                results.push_back(stoch(task, model, loss, criterion, stoch_sga100, header, "stoch-SGA-1.00"));

                results.push_back(stoch(task, model, loss, criterion, stoch_sia050, header, "stoch-SIA-0.50"));
                results.push_back(stoch(task, model, loss, criterion, stoch_sia075, header, "stoch-SIA-0.75"));
                results.push_back(stoch(task, model, loss, criterion, stoch_sia100, header, "stoch-SIA-1.00"));

                // rank algorithms
                std::sort(results.begin(), results.end());

                const thread_pool_t::lock_t lock(mutex);

                const scalar_t best_loss = std::get<0>(*results.begin());

                for (size_t i = 0; i < results.size(); i ++)
                {
                        const scalar_t loss = std::get<0>(results[i]);
                        const string_t name = std::get<1>(results[i]);
                        const size_t miliseconds = std::get<2>(results[i]);

                        giveit(infos, name).update(loss, best_loss, miliseconds);
                }
        });

        std::sort(infos.begin(), infos.end());
        print_all(config_name, infos);
}

int main(int argc, char *argv[])
{
        ncv::init();

        const size_t cmd_samples = 128;
        const size_t cmd_rows = 10;
        const size_t cmd_cols = 10;
        const size_t cmd_outputs = 4;

        syn_dots_task_t task("rows=" + text::to_string(cmd_rows) + "," +
                             "cols=" + text::to_string(cmd_cols) + "," +
                             "color=luma" + "," +
                             "dims=" + text::to_string(cmd_outputs) + "," +
                             "size=" + text::to_string(cmd_samples));
        task.load("");

        const string_t lmodel0;
        const string_t lmodel1 = lmodel0 + "linear:dims=128;act-snorm;";
        const string_t lmodel2 = lmodel1 + "linear:dims=64;act-snorm;";
        const string_t lmodel3 = lmodel2 + "linear:dims=32;act-snorm;";

        string_t cmodel100;
        cmodel100 = cmodel100 + "conv:dims=16,rows=5,cols=5,mask=100;act-snorm;pool-max;";
        cmodel100 = cmodel100 + "conv:dims=32,rows=3,cols=3,mask=100;act-snorm;";

        string_t cmodel50;
        cmodel50 = cmodel50 + "conv:dims=16,rows=5,cols=5,mask=50;act-snorm;pool-max;";
        cmodel50 = cmodel50 + "conv:dims=32,rows=3,cols=3,mask=50;act-snorm;";

        string_t cmodel25;
        cmodel25 = cmodel25 + "conv:dims=16,rows=5,cols=5,mask=25;act-snorm;pool-max;";
        cmodel25 = cmodel25 + "conv:dims=32,rows=3,cols=3,mask=25;act-snorm;";

        const string_t outlayer = "linear:dims=" + text::to_string(cmd_outputs) + ";";

        strings_t cmd_networks =
        {
                lmodel0 + outlayer,
                lmodel1 + outlayer,
                lmodel2 + outlayer,
                lmodel3 + outlayer,

                cmodel100 + outlayer,
                cmodel50 + outlayer,
                cmodel25 + outlayer
        };

        const strings_t cmd_losses = { "square", "cauchy", "logistic" };//loss_manager_t::instance().ids();
        const strings_t cmd_criteria = { "avg" };//criterion_manager_t::instance().ids();

        // vary the model
        for (const string_t& cmd_network : cmd_networks)
        {
                log_info() << "<<< running network [" << cmd_network << "] ...";

                const rmodel_t model = model_manager_t::instance().get("forward-network", cmd_network);
                assert(model);
                model->resize(task, true);

                // vary the loss
                for (const string_t& cmd_loss : cmd_losses)
                {
                        log_info() << "<<< running loss [" << cmd_loss << "] ...";

                        const rloss_t loss = loss_manager_t::instance().get(cmd_loss);
                        assert(loss);

                        // vary the criteria
                        for (const string_t& cmd_criterion : cmd_criteria)
                        {
                                log_info() << "<<< running criterion [" << cmd_criterion << "] ...";

                                test_optimize(task, *model, *loss, cmd_criterion,
                                              "loss [" + cmd_loss + "], criterion [" + cmd_criterion + "]");
                        }
                }

                log_info();
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
