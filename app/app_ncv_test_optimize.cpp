#include "ncv.h"
#include <boost/program_options.hpp>

// display the formatted optimization history
template
<
        typename thistory
>
void print(const ncv::string_t& header, ncv::index_t t, ncv::size_t n,
           const thistory& history)
{
        static const ncv::index_t col_size = 32;

        const ncv::string_t test =
                " [" + ncv::text::to_string(t + 1) + "/" + ncv::text::to_string(n) + "]";

        std::cout << ncv::text::resize(header + test, col_size) << " "
                  << ncv::text::resize("x", col_size) << " "
                  << ncv::text::resize("f(x)", col_size) << " "
                  << ncv::text::resize("|f'(x)|", col_size)
                  << std::endl;

        std::cout << ncv::text::resize(ncv::string_t(col_size, '-'), col_size) << " "
                  << ncv::text::resize(ncv::string_t(col_size, '-'), col_size) << " "
                  << ncv::text::resize(ncv::string_t(col_size, '-'), col_size) << " "
                  << ncv::text::resize(ncv::string_t(col_size, '-'), col_size)
                  << std::endl;

        for (ncv::index_t i = 0; i < history.size(); i ++)
        {
                const ncv::string_t iter =
                        " [" + ncv::text::to_string(i + 1) + "/" + ncv::text::to_string(history.size()) + "]";

                std::cout << ncv::text::resize(header + test + iter, col_size) + "" << " "
                          << ncv::text::resize(ncv::text::vto_string<ncv::scalar_t>(history.x(i)), col_size) << " "
                          << ncv::text::resize(ncv::text::to_string(history.fx(i)), col_size) << " "
                          << ncv::text::resize(ncv::text::to_string(history.gn(i)), col_size)
                          << std::endl;
        }

        std::cout << ncv::text::resize(ncv::string_t(col_size, '-'), col_size) << " "
                  << ncv::text::resize(ncv::string_t(col_size, '-'), col_size) << " "
                  << ncv::text::resize(ncv::string_t(col_size, '-'), col_size) << " "
                  << ncv::text::resize(ncv::string_t(col_size, '-'), col_size)
                  << std::endl;
}

int main(int argc, char *argv[])
{
        typedef ncv::index_t                            index_t;
        typedef ncv::size_t                             size_t;
        typedef ncv::scalar_t                           scalar_t;
        typedef ncv::scalar_vector_t                    vector_t;
        typedef ncv::scalar_vectors_t                   vectors_t;
        typedef ncv::scalar_matrix_t                    matrix_t;
        typedef ncv::optimize::history<scalar_t>        history_t;

        typedef std::function<size_t(void)>                                     op_size_t;
        typedef std::function<scalar_t(const vector_t&)>                        op_fval_t;
        typedef std::function<scalar_t(const vector_t&, vector_t&)>             op_fval_grad_t;
        typedef std::function<scalar_t(const vector_t&, vector_t&, matrix_t&)>  op_fval_grad_hess_t;

        typedef ncv::optimize::problem<
                        scalar_t,
                        op_size_t,
                        op_fval_t,
                        op_fval_grad_t,
                        op_fval_grad_hess_t>            problem_t;

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "help message");
        po_desc.add_options()("iters,i",
                boost::program_options::value<size_t>()->default_value(40),
                "number of iterations [8, 1024]");
        po_desc.add_options()("eps,e",
                boost::program_options::value<scalar_t>()->default_value(1e-6),
                "convergence accuracy [1e-20, 1e-1]");

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

        const size_t cmd_iters = ncv::math::clamp(po_vm["iters"].as<size_t>(), 8, 1024);
        const scalar_t cmd_eps = ncv::math::clamp(po_vm["eps"].as<scalar_t>(), 1e-20, 1e-1);

        // Rosenbrock problem
        {
                const auto rosenbrock_fval = [] (const vector_t& x)
                {
                        return 100.0 * std::pow(x[0] * x[0] - x[1], 2.0) + std::pow(x[0] - 1.0, 2.0);
                };

                const auto rosenbrock_grad = [&] (const vector_t& x, vector_t& g)
                {
                        g.resize(2);
                        g[0] = 400.0 * x[0] * (x[0] * x[0] - x[1]) + 2.0 * (x[0] - 1.0);
                        g[1] = 200.0 * (x[1] - x[0] * x[0]);
                };

                const auto rosenbrock_hess = [&] (const vector_t& x, matrix_t& h)
                {
                        h.resize(2, 2);
                        h(0, 0) = 400.0 * (3.0 * x[0] * x[0] - x[1]) + 2.0;
                        h(0, 1) = h(1, 0) = -400.0 * x[0];
                        h(1, 1) = 200.0;
                };

                const op_size_t rosenbrock_size = [] () -> size_t
                {
                        return 2;
                };

                const op_fval_grad_t rosenbrock_fval_grad = [&] (const vector_t& x, vector_t& g)
                {
                        rosenbrock_grad(x, g);
                        return rosenbrock_fval(x);
                };

                const op_fval_grad_hess_t rosenbrock_fval_grad_hess = [&] (const vector_t& x, vector_t& g, matrix_t& h)
                {
                        rosenbrock_grad(x, g);
                        rosenbrock_hess(x, h);
                        return rosenbrock_fval(x);
                };

                const problem_t problem(rosenbrock_size,
                                        rosenbrock_fval,
                                        rosenbrock_fval_grad,
                                        rosenbrock_fval_grad_hess,
                                        cmd_iters, cmd_eps);

                // intial solutions to test
                vector_t x(2);
                vectors_t x0s;
                {
                        x[0] = -1.0, x[1] = -1.0;
                        x0s.push_back(x);
                }

                // run optimization starting from each initial solution
                for (index_t i = 0; i < x0s.size(); i ++)
                {
                        history_t history;

                        ncv::optimize::gradient_descent(problem, x0s[i], history);
                        print("rosenblock (GD)", i, x0s.size(), history);

                        ncv::optimize::conjugate_gradient_descent(problem, x0s[i], history);
                        print("rosenblock (CGD)", i, x0s.size(), history);

                        ncv::optimize::newton_raphson(problem, x0s[i], history);
                        print("rosenblock (NR)", i, x0s.size(), history);
                }
        }

        // Himmelblau problem
        {
                const auto himmelblau_fval = [] (const vector_t& x)
                {
                        return std::pow(x[0] * x[0] + x[1] - 11.0, 2.0) + std::pow(x[0] + x[1] * x[1] - 7.0, 2.0);
                };

                const auto himmelblau_grad = [&] (const vector_t& x, vector_t& g)
                {
                        g.resize(2);
                        g[0] = 4.0 * x[0] * (x[0] * x[0] + x[1] - 11.0) + 2.0 * (x[0] + x[1] * x[1] - 7.0);
                        g[1] = 2.0 * (x[0] * x[0] + x[1] - 11.0) + 4.0 * x[1] * (x[0] + x[1] * x[1] - 7.0);

                        return himmelblau_fval(x);
                };

                const auto himmelblau_hess = [&] (const vector_t& x, matrix_t& h)
                {
                        h.resize(2, 2);
                        h(0, 0) = 12.0 * x[0] * x[0] + 4.0 * x[1] - 42.0;
                        h(0, 1) = h(1, 0) = 4.0 * x[0] + 4.0 * x[1];
                        h(1, 1) = 12.0 * x[1] * x[1] + 4.0 * x[0] + 2.0;;
                };

                const op_size_t himmelblau_size = [] () -> size_t
                {
                        return 2;
                };

                const op_fval_grad_t himmelblau_fval_grad = [&] (const vector_t& x, vector_t& g)
                {
                        himmelblau_grad(x, g);
                        return himmelblau_fval(x);
                };

                const op_fval_grad_hess_t himmelblau_fval_grad_hess = [&] (const vector_t& x, vector_t& g, matrix_t& h)
                {
                        himmelblau_grad(x, g);
                        himmelblau_hess(x, h);
                        return himmelblau_fval(x);
                };

                const problem_t problem(himmelblau_size,
                                        himmelblau_fval,
                                        himmelblau_fval_grad,
                                        himmelblau_fval_grad_hess,
                                        cmd_iters, cmd_eps);

                // initial solutions to test
                vector_t x(2);
                vectors_t x0s;
                {
                        x[0] = -1.0, x[1] = -1.0;
                        x0s.push_back(x);
                }
                {
                        x[0] = +1.0, x[1] = -1.0;
                        x0s.push_back(x);
                }
                {
                        x[0] = +1.0, x[1] = +1.0;
                        x0s.push_back(x);
                }
                {
                        x[0] = -1.0, x[1] = +1.0;
                        x0s.push_back(x);
                }

                // run optimization starting from each initial solution
                for (index_t i = 0; i < x0s.size(); i ++)
                {
                        history_t history;

                        ncv::optimize::gradient_descent(problem, x0s[i], history);
                        print("himmelblau (GD)", i, x0s.size(), history);

                        ncv::optimize::conjugate_gradient_descent(problem, x0s[i], history);
                        print("himmelblau (CGD)", i, x0s.size(), history);

                        ncv::optimize::newton_raphson(problem, x0s[i], history);
                        print("himmelblau (NR)", i, x0s.size(), history);
                }
        }

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
