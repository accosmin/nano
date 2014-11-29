#include "nanocv.h"
#include "tasks/task_dummy.h"
#include "common/log_search.hpp"
#include <map>

using namespace ncv;

const size_t cmd_iterations = 1024;
const scalar_t cmd_epsilon = 1e-6;
const size_t cmd_trials = 256;

struct opt_info_t
{
        opt_info_t()
                :       m_miliseconds(0),
                        m_iterations(0),
                        m_failures(0),
                        m_notconverged(0),
                        m_count(0)
        {
        }
        
        void update(const opt_state_t& result, size_t max_iterations,  size_t miliseconds)
        {
                m_miliseconds += miliseconds;
                m_iterations += result.m_iterations;
                if (!result.converged(1e-6))
                {
                        if (result.m_iterations < max_iterations)
                        {
                                m_failures ++;
                        }
                        else
                        {
                                m_notconverged ++;
                        }
                }
                m_count ++;                        
        }
        
        size_t  m_miliseconds;          ///< total amount of time (miliseconds)
        size_t  m_iterations;           ///< total number of iterations
        size_t  m_failures;             ///< total number of problems where the algorithm fails
        size_t  m_notconverged;         ///< total number of problems where the algorithm does not converge in the maximum number of iterations
        size_t  m_count;                ///< number of tests
};

std::map<string_t, opt_info_t> opt_statistics;

// display the formatted optimization history (for an optimization problem)
void print_one(const opt_state_t& result, size_t max_iterations, const string_t& header, const string_t& time)
{
        static const size_t col_size = 16;
        static const string_t del_line(4 * col_size + 4, '-');

        std::cout << del_line << std::endl;
        std::cout << header << ": x  = [" << result.x.transpose() << "]" << std::endl;
        std::cout << header << ": fx = [" << result.f << "]" << std::endl;
        std::cout << header << ": gn = [" << result.g.norm() << "]" << std::endl;
        std::cout << header << ": iterations = [" << result.n_iterations() << "/" << max_iterations
                  << "], time = [" << time << "]." << std::endl;
        std::cout << del_line << std::endl;
}

// display the formatted optimization statistics for all optimization algorithms
void print_all()
{
        static const size_t col_size = 16;
        static const string_t del_line(4 * col_size + 4, '$');
        
        std::cout << del_line << std::endl;
        std::cout 
                << text::resize("[algo]", col_size) 
                << text::resize("[failures]", col_size) 
                << text::resize("[not converged]", col_size) 
                << text::resize("[iterations]", col_size)
                << text::resize("[time (ms)]", col_size) << std::endl;
        for (const auto& it : opt_statistics)
        {
                const string_t& name = it.first;
                const opt_info_t& info = it.second;
                
                std::cout 
                        << text::resize(name, col_size) 
                        << text::resize(text::to_string(info.m_failures) + "/" + text::to_string(info.m_count), col_size)  
                        << text::resize(text::to_string(info.m_notconverged) + "/" + text::to_string(info.m_count), col_size)  
                        << text::resize(text::to_string(info.m_iterations), col_size) 
                        << text::resize(text::to_string(info.m_miliseconds), col_size) 
                        << std::endl;
        }
        std::cout << del_line << std::endl;
}

auto fn_wlog = [] (const string_t& message)
{
        log_warning() << message;
};

auto fn_elog = [] (const string_t& message)
{
        log_error() << message;
};

// optimize a problem starting from random points
void test(const opt_problem_t& problem, size_t max_iters, scalar_t eps, const string_t& name, size_t trials)
{
        const size_t size = problem.size();

//        TODO: print rank of methods !!! & #times a method wins
//        size_t ranks[3][3] = { { 0, 0, 0, }, { 0, 0, 0, }, { 0, 0, 0, } };

        random_t<scalar_t> rgen(-2.0, 2.0);
        for (size_t trial = 0; trial < trials; trial ++)
        {
                vector_t x0(size);
                rgen(x0.data(), x0.data() + x0.size());

                const string_t name_trial = " [" + text::to_string(trial + 1) + "/" + text::to_string(trials) + "]";

                #define NCV_TEST_OPTIMIZER(FUN, NAME) \
                { \
                        const ncv::timer_t timer; \
                        problem.reset(); \
                        const opt_state_t res = optimize::FUN(problem, x0, max_iters, eps, fn_wlog, fn_elog); \
                        print_one(res, max_iters, name + " " + #NAME + name_trial, timer.elapsed()); \
                        opt_statistics[#NAME].update(res, max_iters, timer.miliseconds()); \
                }

                NCV_TEST_OPTIMIZER(gd,          GD)
                NCV_TEST_OPTIMIZER(cgd_hs,      CGD-HS)
                NCV_TEST_OPTIMIZER(cgd_fr,      CGD-FR)
                NCV_TEST_OPTIMIZER(cgd_pr,      CGD-PR)
                NCV_TEST_OPTIMIZER(cgd_cd,      CGD-CD)
                NCV_TEST_OPTIMIZER(cgd_ls,      CGD-LS)
                NCV_TEST_OPTIMIZER(cgd_dy,      CGD-DY)
                NCV_TEST_OPTIMIZER(cgd_n,       CGD-N)
                NCV_TEST_OPTIMIZER(lbfgs,       LBFGS)
        }
}

template <typename toptimizer>
void optimize_batch(
        const task_t& task, const model_t& model, const loss_t& loss, const string_t& criterion,
        const toptimizer& optimizer, const string_t& header)
{
        samples_t samples = task.samples();

        accumulator_t ldata(model, ncv::n_threads(), criterion, criterion_t::type::value, 0.1);
        accumulator_t gdata(model, ncv::n_threads(), criterion, criterion_t::type::vgrad, 0.1);

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

        opt_state_t state = optimizer(problem, x0, cmd_iterations, cmd_epsilon,
                                      fn_wlog, fn_elog, fn_ulog);

        log_info() << header << "value = " << state.f << ", done in " << timer.elapsed() << ".";
}

template <typename toptimizer>
void optimize_stoch(
        const task_t& task, const model_t& model, const loss_t& loss, const string_t& criterion,
        const toptimizer& optimizer, const string_t& header)
{
        const size_t cmd_epochs = cmd_iterations;
        const size_t cmd_epoch_size = task.samples().size();
        const scalar_t cmd_beta = std::pow(0.01, 1.0 / (cmd_epochs * cmd_epoch_size));

        const ncv::timer_t timer;

        // tune the learning rate
        const auto op_tune_alpha0 = [&] (scalar_t alpha0)
        {
                samples_t samples = task.samples();

                accumulator_t ldata(model, ncv::n_threads(), criterion, criterion_t::type::value, 0.1);
                accumulator_t gdata(model, ncv::n_threads(), criterion, criterion_t::type::vgrad, 0.1);

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

                opt_state_t state = optimizer(problem, x0, cmd_epochs, cmd_epoch_size, alpha0, cmd_beta,
                                              fn_ulog);

                ldata.reset(state.x);
                ldata.update(task, samples, loss);
                state.f = ldata.value();        // need to compute the cumulative loss for stochastic optimizers

                return state;
        };

        ncv::thread_pool_t wpool;
        const opt_state_t state = ncv::log_min_search_mt(op_tune_alpha0, wpool, -6.0, -1.0, 0.5, ncv::n_threads());

        log_info() << header << "value = " << state.f << ", done in " << timer.elapsed() << ".";
}

void test_optimize(const task_t& task, model_t& model, const loss_t& loss, const string_t& criterion)
{
        for (size_t cmd_trial = 0; cmd_trial < cmd_trials; cmd_trial ++)
        {
                model.random_params();

                const string_t header = "[" + text::to_string(cmd_trial) + "/" + text::to_string(cmd_trials) + "] ";

                // batch optimizers
                optimize_batch(task, model, loss, criterion, optimize::gd<opt_problem_t>, header + "[batch-GD]: ");
                optimize_batch(task, model, loss, criterion, optimize::cgd_n<opt_problem_t>, header + "[batch-CGD-N]: ");
                optimize_batch(task, model, loss, criterion, optimize::cgd_cd<opt_problem_t>, header + "[batch-CGD-CD]: ");
                optimize_batch(task, model, loss, criterion, optimize::cgd_dy<opt_problem_t>, header + "[batch-CGD-DY]: ");
                optimize_batch(task, model, loss, criterion, optimize::cgd_fr<opt_problem_t>, header + "[batch-CGD-FR]: ");
                optimize_batch(task, model, loss, criterion, optimize::cgd_hs<opt_problem_t>, header + "[batch-CGD-HS]: ");
                optimize_batch(task, model, loss, criterion, optimize::cgd_ls<opt_problem_t>, header + "[batch-CGD-LS]: ");
                optimize_batch(task, model, loss, criterion, optimize::cgd_pr<opt_problem_t>, header + "[batch-CGD-PR]: ");
                optimize_batch(task, model, loss, criterion, optimize::lbfgs<opt_problem_t>, header + "[batch-LBFGS]: ");

                // stochastic optimizers
                optimize_stoch(task, model, loss, criterion, optimize::stoch_sg<opt_problem_t>, header + "[stoch-SG]: ");
                optimize_stoch(task, model, loss, criterion, optimize::stoch_sga<opt_problem_t>, header + "[stoch-SGA]: ");
                optimize_stoch(task, model, loss, criterion, optimize::stoch_sia<opt_problem_t>, header + "[stoch-SIA]: ");
        }
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

        const strings_t cmd_losses = loss_manager_t::instance().ids();
        const strings_t cmd_criteria = criterion_manager_t::instance().ids();

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

                                test_optimize(task, *model, *loss, cmd_criterion);
                        }
                }

                log_info();
        }


        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
