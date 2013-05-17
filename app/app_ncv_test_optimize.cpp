#include "ncv.h"
#include <boost/program_options.hpp>

// display the formatted optimization history
template
<
        typename tproblem
>
void print(const tproblem& problem, const ncv::string_t& header, const ncv::string_t& time)
{
        static const ncv::size_t col_size = 32;
        static const ncv::string_t del_line(4 * col_size + 4, '-');

        std::cout << del_line << std::endl;
        std::cout << header << ": x  = [" << problem.optimum().x.transpose() << "]" << std::endl;
        std::cout << header << ": fx = [" << problem.optimum().f << "]" << std::endl;
        std::cout << header << ": gn = [" << problem.optimum().g.norm() << "]" << std::endl;
        std::cout << header << ": evaluations = [" << problem.fevals() << " + " << problem.gevals()
                  << "], iterations = [" << problem.iterations() << "/" << problem.max_iterations()
                  << "], speed = [" << problem.speed_avg() << " +/- " << problem.speed_stdev()
                  << "], time = [" << time << "]." << std::endl;
        std::cout << del_line << std::endl;
}

// optimize a problem starting from random points
template
<
        typename tproblem
>
void test(const tproblem& problem, const ncv::string_t& name, ncv::size_t trials)
{
        const ncv::size_t size = problem.size();

        ncv::random_t<ncv::scalar_t> rgen(-1.0, 1.0);
        for (size_t trial = 0; trial < trials; trial ++)
        {
                ncv::vector_t x0(size);
                rgen(x0.data(), x0.data() + x0.size());

                const ncv::string_t name_trial =
                        " [" + ncv::text::to_string(trial + 1) + "/" + ncv::text::to_string(trials) + "]";

                ncv::timer_t timer;

                timer.start();
                ncv::optimize::gradient_descent(problem, x0);
                print(problem, name + " (GD)" + name_trial, timer.elapsed_string());

                timer.start();
                ncv::optimize::conjugate_gradient_descent(problem, x0);
                print(problem, name + " (CGD)" + name_trial, timer.elapsed_string());

                timer.start();
                ncv::optimize::lbfgs(problem, x0);
                print(problem, name + " (LBFGS)" + name_trial, timer.elapsed_string());
        }
}

int main(int argc, char *argv[])
{
        typedef ncv::size_t                             size_t;
        typedef ncv::scalar_t                           scalar_t;
        typedef ncv::vector_t                           vector_t;

        typedef std::function<size_t(void)>                                     op_size_t;
        typedef std::function<scalar_t(const vector_t&)>                        op_fval_t;
        typedef std::function<scalar_t(const vector_t&, vector_t&)>             op_fval_grad_t;

        typedef ncv::optimize::problem_t<
                        op_size_t,
                        op_fval_t,
                        op_fval_grad_t>                 problem_t;

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

        const size_t cmd_iters = ncv::math::clamp(po_vm["iters"].as<size_t>(), 8, 16000);
        const scalar_t cmd_eps = ncv::math::clamp(po_vm["eps"].as<scalar_t>(), 1e-20, 1e-1);
        const size_t cmd_dims = ncv::math::clamp(po_vm["dim"].as<size_t>(), 2, 1024);
        const size_t cmd_trials = 16;

//        // sphere function
//        for (size_t n = 2; n <= cmd_dims; n *= 2)
//        {
//                const auto op_size = [=] ()
//                {
//                        return n;
//                };

//                const auto op_fval = [=] (const vector_t& x)
//                {
//                        return x.dot(x);
//                };

//                const auto op_grad = [=] (const vector_t& x, vector_t& g)
//                {
//                        g = 2.0 * x;
//                };

//                const auto op_fval_grad = [=] (const vector_t& x, vector_t& g)
//                {
//                        op_grad(x, g);
//                        return op_fval(x);
//                };

//                const problem_t problem(op_size, op_fval, op_fval_grad, cmd_iters, cmd_eps);
//                test(problem, "sphere [" + ncv::text::to_string(n) + "D]", cmd_trials);
//        }

//        // ellipsoidal function
//        for (size_t n = 2; n <= cmd_dims; n *= 2)
//        {
//                const auto op_size = [=] ()
//                {
//                        return n;
//                };

//                const auto op_fval = [=] (const vector_t& x)
//                {
//                        scalar_t f = 0.0;
//                        for (size_t i = 0; i < n; i ++)
//                        {
//                                f += (i + 1.0) * ncv::math::square(x[i]);
//                        }
//                        return f;
//                };

//                const auto op_grad = [=] (const vector_t& x, vector_t& g)
//                {
//                        g.resize(n);
//                        for (size_t i = 0; i < n; i ++)
//                        {
//                                g(i) = 2.0 * (i + 1.0) * x[i];
//                        }
//                };

//                const auto op_fval_grad = [=] (const vector_t& x, vector_t& g)
//                {
//                        op_grad(x, g);
//                        return op_fval(x);
//                };

//                const problem_t problem(op_size, op_fval, op_fval_grad, cmd_iters, cmd_eps);
//                test(problem, "ellipsoidal [" + ncv::text::to_string(n) + "D]", cmd_trials);
//        }

//        // rotated ellipsoidal function
//        for (size_t n = 2; n <= cmd_dims; n *= 2)
//        {
//                const auto op_size = [=] ()
//                {
//                        return n;
//                };

//                const auto op_fval = [=] (const vector_t& x)
//                {
//                        scalar_t f = 0.0;

//                        for (size_t i = 0; i < n; i ++)
//                        {
//                                scalar_t s = 0.0;
//                                for (size_t j = 0; j <= i; j ++)
//                                {
//                                        s += x[j];
//                                }

//                                f += ncv::math::square(s);
//                        }
//                        return f;
//                };

//                const problem_t problem(op_size, op_fval, cmd_iters, cmd_eps);
//                test(problem, "rotated ellipsoidal [" + ncv::text::to_string(n) + "D]", cmd_trials);
//        }

//        // Whitley's function
//        for (size_t n = 2; n <= cmd_dims; n *= 2)
//        {
//                const auto op_size = [=] ()
//                {
//                        return n;
//                };

//                const auto op_fval = [=] (const vector_t& x)
//                {
//                        scalar_t f = 0.0;
//                        for (size_t i = 0; i < n; i ++)
//                        {
//                                for (size_t j = 0; j < n; j ++)
//                                {
//                                        const scalar_t d = 100.0 * ncv::math::square(x[i] * x[i] - x[j]) +
//                                                           ncv::math::square(1.0 - x[j]);
//                                        f += d * d / 4000.0 - std::cos(d) + 1.0;
//                                }
//                        }

//                        return f;
//                };

//                const problem_t problem(op_size, op_fval, cmd_iters, cmd_eps);
//                test(problem, "whitley [" + ncv::text::to_string(n) + "D]", cmd_trials);
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
                                f += 100.0 * ncv::math::square(x[i + 1] - ncv::math::square(x[i])) +
                                     ncv::math::square(x[i] - 1.0);
                        }

                        return f;
                };

                const problem_t problem(op_size, op_fval, cmd_iters, cmd_eps);
                test(problem, "rosenbrock [" + ncv::text::to_string(n) + "D]", cmd_trials);
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

                const problem_t problem(op_size, op_fval, op_fval_grad, cmd_iters, cmd_eps);
                test(problem, "himmelblau [2D]", cmd_trials);
        }

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
