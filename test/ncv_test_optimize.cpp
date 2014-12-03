#include "nanocv.h"
#include "tasks/task_dummy.h"
#include "common/log_search.hpp"
#include <map>

using namespace ncv;

const size_t cmd_iterations = 128;
const scalar_t cmd_epsilon = 1e-6;
const size_t cmd_trials = 128;

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
        static const size_t col_size = 24;
        static const string_t del_line(3 * col_size, '$');
        
        std::cout << del_line << std::endl;
        std::cout << text::resize(name, del_line.size(), align::center, '$') << std::endl;
        std::cout << del_line << std::endl;
        std::cout << text::resize("[algorithm]", col_size);
        std::cout << text::resize("[cumulated score]", col_size);
        std::cout << text::resize("[total time (ms)]", col_size) << std::endl;

        for (const opt_info_t& info : infos)
        {
                std::cout << text::resize(info.m_name, col_size);
                std::cout << text::resize(text::to_string(info.m_score), col_size);
                std::cout << text::resize(text::to_string(info.m_miliseconds), col_size);
                std::cout << std::endl;
        }
        std::cout << del_line << std::endl;
}

template <typename toptimizer>
std::tuple<scalar_t, string_t, size_t> batch(
        const task_t& task, const model_t& model, const loss_t& loss, const string_t& criterion,
        const toptimizer& optimizer, const string_t& header, const string_t& name)
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

        // assembly optimization problem & optimize the model
        const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

        const opt_state_t state = optimizer(problem, x0, cmd_iterations, cmd_epsilon, nullptr, nullptr, nullptr);

//        log_info() << name << ": value = " << state.f << ", done in " << timer.elapsed() << ".";

        // OK
        return std::make_tuple(state.f, name, timer.miliseconds());
}

template <typename toptimizer>
std::tuple<scalar_t, string_t, size_t> stoch(
        const task_t& task, const model_t& model, const loss_t& loss, const string_t& criterion,
        const toptimizer& optimizer, const string_t& header, const string_t& name)
{
        const size_t cmd_epochs = cmd_iterations;
        const size_t cmd_epoch_size = task.samples().size();

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

                opt_state_t state = optimizer(problem, x0, cmd_epochs, cmd_epoch_size, alpha0, fn_ulog);

                ldata.reset(state.x);
                ldata.update(task, samples, loss);
                state.f = ldata.value();        // need to compute the cumulative loss for stochastic optimizers

                return state;
        };

        const opt_state_t state = ncv::log_min_search(op_tune_alpha0, -6.0, -1.0, 0.5, 4);

//        log_info() << header << "[" + name << "]: value = " << state.f << ", done in " << timer.elapsed() << ".";

        // OK
        return std::make_tuple(state.f, name, timer.miliseconds());
}

void test_optimize(
        const task_t& task, const model_t& imodel, const loss_t& loss, const string_t& criterion,
        const string_t& config_name)
{
        opt_infos_t infos;

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
                results.push_back(batch(task, model, loss, criterion, optimize::gd<opt_problem_t>, header, "batch-GD"));
                results.push_back(batch(task, model, loss, criterion, optimize::cgd_n<opt_problem_t>, header, "batch-CGD-N"));
                results.push_back(batch(task, model, loss, criterion, optimize::cgd_cd<opt_problem_t>, header, "batch-CGD-CD"));
                results.push_back(batch(task, model, loss, criterion, optimize::cgd_dy<opt_problem_t>, header, "batch-CGD-DY"));
                results.push_back(batch(task, model, loss, criterion, optimize::cgd_fr<opt_problem_t>, header, "batch-CGD-FR"));
                results.push_back(batch(task, model, loss, criterion, optimize::cgd_hs<opt_problem_t>, header, "batch-CGD-HS"));
                results.push_back(batch(task, model, loss, criterion, optimize::cgd_ls<opt_problem_t>, header, "batch-CGD-LS"));
                results.push_back(batch(task, model, loss, criterion, optimize::cgd_pr<opt_problem_t>, header, "batch-CGD-PR"));
                results.push_back(batch(task, model, loss, criterion, optimize::lbfgs<opt_problem_t>, header, "batch-LBFGS"));

                // stochastic optimizers
                results.push_back(stoch(task, model, loss, criterion, optimize::stoch_nag<opt_problem_t>, header, "stoch-NAG"));

                results.push_back(stoch(task, model, loss, criterion, optimize::stoch_sg<optimize::decay_rate::sqrt, opt_problem_t>, header, "stoch-SG-SQRT"));
                results.push_back(stoch(task, model, loss, criterion, optimize::stoch_sg<optimize::decay_rate::qrt3, opt_problem_t>, header, "stoch-SG-QRT3"));
                results.push_back(stoch(task, model, loss, criterion, optimize::stoch_sg<optimize::decay_rate::unit, opt_problem_t>, header, "stoch-SG-UNIT"));

                results.push_back(stoch(task, model, loss, criterion, optimize::stoch_sga<optimize::decay_rate::sqrt, opt_problem_t>, header, "stoch-SGA-SQRT"));
                results.push_back(stoch(task, model, loss, criterion, optimize::stoch_sga<optimize::decay_rate::qrt3, opt_problem_t>, header, "stoch-SGA-QRT3"));
                results.push_back(stoch(task, model, loss, criterion, optimize::stoch_sga<optimize::decay_rate::unit, opt_problem_t>, header, "stoch-SGA-UNIT"));

                results.push_back(stoch(task, model, loss, criterion, optimize::stoch_sia<optimize::decay_rate::sqrt, opt_problem_t>, header, "stoch-SIA-SQRT"));
                results.push_back(stoch(task, model, loss, criterion, optimize::stoch_sia<optimize::decay_rate::qrt3, opt_problem_t>, header, "stoch-SIA-QRT3"));
                results.push_back(stoch(task, model, loss, criterion, optimize::stoch_sia<optimize::decay_rate::unit, opt_problem_t>, header, "stoch-SIA-UNIT"));

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
