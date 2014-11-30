#include "nanocv.h"
#include "tasks/task_dummy.h"
#include "common/log_search.hpp"
#include <map>

using namespace ncv;

const size_t cmd_iterations = 1024;
const scalar_t cmd_epsilon = 1e-6;
const size_t cmd_trials = 16;
const size_t n_algorithms = 13;

// optimization statistics for a particular algorithm
struct opt_info_t
{
        opt_info_t() :  m_miliseconds(0),
                        m_ranks(n_algorithms, 0)
        {
        }
        
        void update(size_t rank, size_t miliseconds)
        {
                assert(rank < n_algorithms);

                m_miliseconds += miliseconds;
                m_ranks[rank] ++;
        }
        
        std::size_t             m_miliseconds;          ///< total amount of time (miliseconds)
        std::vector<size_t>     m_ranks;                ///< total number of times per rank
};

typedef std::map<string_t, opt_info_t>  opt_infos_t;

// display the formatted optimization statistics for all optimization algorithms
void print_all(const opt_infos_t& infos)
{
        static const size_t col_size = 10;
        static const string_t del_line((2 + n_algorithms) * col_size + 12, '$');
        
        std::cout << del_line << std::endl;
        std::cout << text::resize("[algorithm]", col_size * 2);
        for (size_t i = 0; i < n_algorithms; i ++)
        {
                std::cout << text::resize("[rank" + text::to_string(i + 1) + "]", col_size);
        }
        std::cout << text::resize("[time (ms)]", col_size) << std::endl;

        for (const auto& it : infos)
        {
                const string_t& name = it.first;
                const opt_info_t& info = it.second;
                
                std::cout << text::resize(name, col_size * 2);
                for (size_t i = 0; i < n_algorithms; i ++)
                {
                        std::cout << text::resize(text::to_string(info.m_ranks[i]), col_size);
                }
                std::cout << text::resize(text::to_string(info.m_miliseconds), col_size);
                std::cout << std::endl;
        }
        std::cout << del_line << std::endl;
}

template <typename toptimizer>
void optimize_batch(
        const task_t& task, const model_t& model, const loss_t& loss, const string_t& criterion,
        const toptimizer& optimizer, const string_t& header, const string_t& name,
        std::vector<std::tuple<scalar_t, string_t, size_t>>& results)
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

        auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
        {
                gdata.reset(x);
                gdata.update(task, samples, loss);

                gx = gdata.vgrad();
                return gdata.value();
        };

        auto fn_wlog = [] (const string_t& message)
        {
                log_warning() << message;
        };
        auto fn_elog = [] (const string_t& message)
        {
                log_error() << message;
        };
        const opt_opulog_t fn_ulog = [&] (const opt_state_t&)
        {
        };

        // assembly optimization problem & optimize the model
        const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

        const opt_state_t state = optimizer(problem, x0, cmd_iterations, cmd_epsilon, fn_wlog, fn_elog, fn_ulog);

//        log_info() << header << "[" + name << "]: value = " << state.f << ", done in " << timer.elapsed() << ".";

        results.push_back(std::make_tuple(state.f, name, timer.miliseconds()));
}

template <typename toptimizer>
void optimize_stoch(
        const task_t& task, const model_t& model, const loss_t& loss, const string_t& criterion,
        const toptimizer& optimizer, const string_t& header, const string_t& name,
        std::vector<std::tuple<scalar_t, string_t, size_t>>& results)
{
        const size_t cmd_epochs = cmd_iterations;
        const size_t cmd_epoch_size = task.samples().size();
        const scalar_t cmd_beta = std::pow(0.01, 1.0 / (cmd_epochs * cmd_epoch_size));

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

                auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
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
                const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

                opt_state_t state = optimizer(problem, x0, cmd_epochs, cmd_epoch_size, alpha0, cmd_beta, fn_ulog);

                ldata.reset(state.x);
                ldata.update(task, samples, loss);
                state.f = ldata.value();        // need to compute the cumulative loss for stochastic optimizers

                return state;
        };

        const opt_state_t state = ncv::log_min_search(op_tune_alpha0, -6.0, -1.0, 0.5, 4);

//        log_info() << header << "[" + name << "]: value = " << state.f << ", done in " << timer.elapsed() << ".";

        results.push_back(std::make_tuple(state.f, name, timer.miliseconds()));
}

void test_optimize(const task_t& task, const model_t& imodel, const loss_t& loss, const string_t& criterion,
        opt_infos_t& infos)
{
        thread_pool_t::mutex_t mutex;

        ncv::thread_loopi(cmd_trials, [&] (size_t cmd_trial)
        {
                // random initialization
                const rmodel_t rmodel = imodel.clone();
                model_t& model = *rmodel;
                model.random_params();

                const string_t header = "[" + text::to_string(cmd_trial) + "/" + text::to_string(cmd_trials) + "] ";

                std::vector<std::tuple<scalar_t, string_t, size_t>> results;

                // batch optimizers
                optimize_batch(task, model, loss, criterion, optimize::gd<opt_problem_t>, header, "batch-GD", results);
                optimize_batch(task, model, loss, criterion, optimize::cgd_n<opt_problem_t>, header, "batch-CGD-N", results);
                optimize_batch(task, model, loss, criterion, optimize::cgd_cd<opt_problem_t>, header, "batch-CGD-CD", results);
                optimize_batch(task, model, loss, criterion, optimize::cgd_dy<opt_problem_t>, header, "batch-CGD-DY", results);
                optimize_batch(task, model, loss, criterion, optimize::cgd_fr<opt_problem_t>, header, "batch-CGD-FR", results);
                optimize_batch(task, model, loss, criterion, optimize::cgd_hs<opt_problem_t>, header, "batch-CGD-HS", results);
                optimize_batch(task, model, loss, criterion, optimize::cgd_ls<opt_problem_t>, header, "batch-CGD-LS", results);
                optimize_batch(task, model, loss, criterion, optimize::cgd_pr<opt_problem_t>, header, "batch-CGD-PR", results);
                optimize_batch(task, model, loss, criterion, optimize::lbfgs<opt_problem_t>, header, "batch-LBFGS", results);

                // stochastic optimizers
                optimize_stoch(task, model, loss, criterion, optimize::stoch_sg<opt_problem_t>, header, "stoch-SG", results);
                optimize_stoch(task, model, loss, criterion, optimize::stoch_sga<opt_problem_t>, header, "stoch-SGA", results);
                optimize_stoch(task, model, loss, criterion, optimize::stoch_sia<opt_problem_t>, header, "stoch-SIA", results);
                optimize_stoch(task, model, loss, criterion, optimize::stoch_nag<opt_problem_t>, header, "stoch-NAG", results);

                // rank algorithms
                std::sort(results.begin(), results.end());

                const thread_pool_t::lock_t lock(mutex);

                for (size_t rank = 0; rank < results.size(); rank ++)
                {
                        const string_t& name = std::get<1>(results[rank]);
                        const size_t miliseconds = std::get<2>(results[rank]);

                        infos[name].update(rank, miliseconds);
                }
        });
}

void test_optimize(const task_t& task, model_t& model)
{
        opt_infos_t infos;

        const strings_t cmd_losses = loss_manager_t::instance().ids();
        const strings_t cmd_criteria = criterion_manager_t::instance().ids();

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

                        test_optimize(task, model, *loss, cmd_criterion, infos);
                }
        }

        print_all(infos);
}

int main(int argc, char *argv[])
{
        ncv::init();

        const size_t cmd_samples = 128;
        const size_t cmd_rows = 10;
        const size_t cmd_cols = 10;
        const size_t cmd_outputs = 4;

        dummy_task_t task;
        task.set_rows(cmd_rows);
        task.set_cols(cmd_cols);
        task.set_color(color_mode::luma);
        task.set_outputs(cmd_outputs);
        task.set_folds(1);
        task.set_size(cmd_samples);
        task.setup();

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

        // vary the model
        for (const string_t& cmd_network : cmd_networks)
        {
                log_info() << "<<< running network [" << cmd_network << "] ...";

                const rmodel_t model = model_manager_t::instance().get("forward-network", cmd_network);
                assert(model);
                model->resize(task, true);

                test_optimize(task, *model);

                log_info();
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
