#include "timer.h"
#include "measure.h"
#include "text/table.h"
#include "text/cmdline.h"
#include "math/stats.h"
#include "functions/test.h"
#include <iostream>

using namespace nano;

static void eval_func(const function_t& function, table_t& table)
{
        stats_t<scalar_t> fval_times;
        stats_t<scalar_t> grad_times;

        const auto dims = function.size();
        const vector_t x = vector_t::Zero(dims);
        vector_t g = vector_t::Zero(dims);

        const size_t trials = 16;

        volatile scalar_t fx = 0;
        const auto fval_time = measure_robustly<nanoseconds_t>([&] ()
        {
                fx += function.eval(x);
        }, trials).count();

        volatile scalar_t gx = 0;
        const auto grad_time = measure_robustly<nanoseconds_t>([&] ()
        {
                function.eval(x, &g);
                gx += g.template lpNorm<Eigen::Infinity>();
        }, trials).count();

        auto& row = table.append();
        row << function.name() << fval_time << grad_time;
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

        table_t table;
        table.header() << "function" << "f(x) [ns]" << "f(x, g) [ns]";

        foreach_test_function(make_functions(min_dims, max_dims), [&] (const function_t& function)
        {
                eval_func(function, table);
        });

        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
