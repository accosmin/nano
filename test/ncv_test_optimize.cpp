#include "nanocv.h"
#include <boost/program_options.hpp>
#include <map>

using namespace ncv;

struct opt_info_t
{
        opt_info_t()
                :       m_miliseconds(0),
                        m_iterations(0),
                        m_failures(0),
                        m_count(0)
        {
        }
        
        void update(const opt_state_t& result, size_t miliseconds)
        {
                m_miliseconds += miliseconds;
                m_iterations += result.m_iterations;
                if (!result.converged(1e-6))
                {
                        m_failures ++;
                }
                m_count ++;                        
        }
        
        size_t  m_miliseconds;          ///< total amount of time (miliseconds)
        size_t  m_iterations;           ///< total number of iterations
        size_t  m_failures;             ///< total number of failed problems
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
        static const size_t col_size = 32;
        static const string_t del_line(4 * col_size + 4, '$');
        
        std::cout << del_line << std::endl;
        std::cout 
                << ncv::text::resize("[algo]", col_size) 
                << ncv::text::resize("[failures]", col_size) 
                << ncv::text::resize("[iterations]", col_size)
                << ncv::text::resize("[time (ms)]", col_size) << std::endl;
        for (const auto& it : opt_statistics)
        {
                const string_t& name = it.first;
                const opt_info_t& info = it.second;
                
                std::cout 
                        << ncv::text::resize(name, col_size) 
                        << ncv::text::resize(ncv::text::to_string(info.m_failures), col_size)  
                        << ncv::text::resize(ncv::text::to_string(info.m_iterations), col_size) 
                        << ncv::text::resize(ncv::text::to_string(info.m_miliseconds), col_size) 
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
                        opt_statistics[#NAME].update(res, timer.miliseconds()); \
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

int main(int argc, char *argv[])
{
        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "help message");
        po_desc.add_options()("iters",
                boost::program_options::value<size_t>()->default_value(2048),
                "number of iterations [8, 16000]");
        po_desc.add_options()("eps",
                boost::program_options::value<scalar_t>()->default_value(1e-6),
                "convergence accuracy [1e-20, 1e-1]");
        po_desc.add_options()("dim",
                boost::program_options::value<size_t>()->default_value(128),
                "maximum dimension [2, 1024]");

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

        const size_t cmd_iters = math::clamp(po_vm["iters"].as<size_t>(), 8, 16000);
        const scalar_t cmd_eps = math::clamp(po_vm["eps"].as<scalar_t>(), 1e-20, 1e-1);
        const size_t cmd_dims = math::clamp(po_vm["dim"].as<size_t>(), 2, 1024);
        const size_t cmd_trials = 64;

//         // sphere function
//         for (size_t n = 2; n <= cmd_dims; n *= 2)
//         {
//                 const auto op_size = [=] ()
//                 {
//                         return n;
//                 };
// 
//                 const auto op_fval = [=] (const vector_t& x)
//                 {
//                         return x.dot(x);
//                 };
// 
//                 const auto op_grad = [=] (const vector_t& x, vector_t& g)
//                 {
//                         g = 2.0 * x;
//                 };
// 
//                 const auto op_fval_grad = [=] (const vector_t& x, vector_t& g)
//                 {
//                         op_grad(x, g);
//                         return op_fval(x);
//                 };
// 
//                 const opt_problem_t problem(op_size, op_fval, op_fval_grad);
//                 test(problem, cmd_iters, cmd_eps, "sphere [" + text::to_string(n) + "D]", cmd_trials);
//         }

//         // ellipsoidal function
//         for (size_t n = 2; n <= cmd_dims; n *= 2)
//         {
//                 const auto op_size = [=] ()
//                 {
//                         return n;
//                 };
// 
//                 const auto op_fval = [=] (const vector_t& x)
//                 {
//                         scalar_t f = 0.0;
//                         for (size_t i = 0; i < n; i ++)
//                         {
//                                 f += (i + 1.0) * math::square(x[i]);
//                         }
//                         return f;
//                 };
// 
//                const auto op_grad = [=] (const vector_t& x, vector_t& g)
//                {
//                        g.resize(n);
//                        for (size_t i = 0; i < n; i ++)
//                        {
//                                g(i) = 2.0 * (i + 1.0) * x[i];
//                        }
//                };
// 
//                const auto op_fval_grad = [=] (const vector_t& x, vector_t& g)
//                {
//                        op_grad(x, g);
//                        return op_fval(x);
//                };
// 
//                const opt_problem_t problem(op_size, op_fval, op_fval_grad);
//                test(problem, cmd_iters, cmd_eps, "ellipsoidal [" + text::to_string(n) + "D]", cmd_trials);
//        }

//        // rotated ellipsoidal function
//        for (size_t n = 2; n <= cmd_dims; n *= 2)
//        {
//                const auto op_size = [=] ()
//                {
//                        return n;
//                };
// 
//                const auto op_fval = [=] (const vector_t& x)
//                {
//                        scalar_t f = 0.0;
// 
//                        for (size_t i = 0; i < n; i ++)
//                        {
//                                scalar_t s = 0.0;
//                                for (size_t j = 0; j <= i; j ++)
//                                {
//                                        s += x[j];
//                                }
// 
//                                f += math::square(s);
//                        }
//                        return f;
//                };
// 
//                const opt_problem_t problem(op_size, op_fval);
//                test(problem, cmd_iters, cmd_eps, "rotated ellipsoidal [" + text::to_string(n) + "D]", cmd_trials);
//        }
// 
//        // Whitley's function
//        for (size_t n = 2; n <= cmd_dims; n *= 2)
//        {
//                const auto op_size = [=] ()
//                {
//                        return n;
//                };
// 
//                const auto op_fval = [=] (const vector_t& x)
//                {
//                        scalar_t f = 0.0;
//                        for (size_t i = 0; i < n; i ++)
//                        {
//                                for (size_t j = 0; j < n; j ++)
//                                {
//                                        const scalar_t d = 100.0 * math::square(x[i] * x[i] - x[j]) +
//                                                           math::square(1.0 - x[j]);
//                                        f += d * d / 4000.0 - std::cos(d) + 1.0;
//                                }
//                        }
// 
//                        return f;
//                };
// 
//                const opt_problem_t problem(op_size, op_fval);
//                test(problem, cmd_iters, cmd_eps, "whitley [" + text::to_string(n) + "D]", cmd_trials);
//        }

        // Rosenbrock problem
        for (size_t n = 2; n <= cmd_dims; n *= 2)
        {
                const auto op_size = [=] ()
                {
                        return n;
                };

                const auto op_fval = [=] (const vector_t& x)
                {
                        scalar_t f = 0.0;
                        for (size_t i = 0; i + 1 < n; i ++)
                        {
                                f += 100.0 * math::square(x[i + 1] - math::square(x[i])) +
                                     math::square(x[i] - 1.0);
                        }

                        return f;
                };

                const opt_problem_t problem(op_size, op_fval);
                test(problem, cmd_iters, cmd_eps, "rosenbrock [" + text::to_string(n) + "D]", cmd_trials);
        }

        // Himmelblau problem
        {
                const auto op_size = [=] ()
                {
                        return 2;
                };

                const auto op_fval = [=] (const vector_t& x)
                {
                        return std::pow(x[0] * x[0] + x[1] - 11.0, 2.0) + std::pow(x[0] + x[1] * x[1] - 7.0, 2.0);
                };

                const auto op_grad = [=] (const vector_t& x, vector_t& g)
                {
                        g.resize(2);
                        g[0] = 4.0 * x[0] * (x[0] * x[0] + x[1] - 11.0) + 2.0 * (x[0] + x[1] * x[1] - 7.0);
                        g[1] = 2.0 * (x[0] * x[0] + x[1] - 11.0) + 4.0 * x[1] * (x[0] + x[1] * x[1] - 7.0);
                };

                const auto op_fval_grad = [=] (const vector_t& x, vector_t& g)
                {
                        op_grad(x, g);
                        return op_fval(x);
                };

                const opt_problem_t problem(op_size, op_fval, op_fval_grad);
                test(problem, cmd_iters, cmd_eps, "himmelblau [2D]", cmd_trials);
        }
        
        print_all();

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
