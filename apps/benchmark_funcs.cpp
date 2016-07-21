#include "timer.h"
#include "logger.h"
#include "measure.hpp"
#include "text/table.h"
#include "optim/test.h"
#include "text/cmdline.h"
#include "math/stats.hpp"
#include <iostream>

using namespace nano;

static void eval_func(const function_t& function, table_t& table)
{
        const auto problem = function.problem();

        stats_t<scalar_t> fval_times;
        stats_t<scalar_t> grad_times;

        const auto dims = function.problem().size();
        const vector_t x = vector_t::Zero(dims);
        vector_t g = vector_t::Zero(dims);

        const size_t trials = 16;

        scalar_t fx = 0;
        const auto fval_time = measure_robustly_nsec([&] ()
        {
                fx += problem(x);
        }, trials).count();

        scalar_t gx = 0;
        const auto grad_time = measure_robustly_nsec([&] ()
        {
                problem(x, g);
                gx += g.template lpNorm<Eigen::Infinity>();
        }, trials).count();

        auto& row = table.append(function.name());
        row << fval_time << grad_time;
}

int main(int argc, const char* argv[])
{
        // parse the command line
        cmdline_t cmdline("benchmark optimization test functions");
        cmdline.add("", "min-dims", "minimum number of dimensions for each test function (if feasible)", "128");
        cmdline.add("", "max-dims", "maximum number of dimensions for each test function (if feasible)", "1024");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_dims = cmdline.get<tensor_size_t>("min-dims");
        const auto max_dims = cmdline.get<tensor_size_t>("max-dims");

        table_t table("function");
        table.header() << "f(x) [ns]" << "f(x, g) [ns]";

        foreach_test_function(make_functions(min_dims, max_dims), [&] (const function_t& function)
        {
                eval_func(function, table);
        });

        table.print(std::cout);

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

