#include "tensor.h"
#include "text/table.h"
#include "text/cmdline.h"
#include "math/stats.hpp"
#include "cortex/timer.h"
#include "cortex/logger.h"
#include "cortex/measure.hpp"
#include "math/funcs/foreach.hpp"
#include <iostream>

namespace
{
        using namespace nano;

        template
        <
                typename tscalar,
                typename tsize = typename nano::function_t<tscalar>::tsize,
                typename tvector = typename nano::function_t<tscalar>::tvector,
                typename tproblem = typename nano::function_t<tscalar>::tproblem
        >
        void eval_func(const nano::function_t<tscalar>& function, nano::table_t& table)
        {
                const auto problem = function.problem();

                nano::stats_t<scalar_t> fval_times;
                nano::stats_t<scalar_t> grad_times;

                const auto dims = function.problem().size();
                const tvector x = tvector::Zero(dims);
                tvector g = tvector::Zero(dims);

                const size_t trials = 16;

                tscalar fx = 0;
                const auto fval_time = nano::measure_robustly_nsec([&] ()
                {
                        fx += problem(x);
                }, trials).count();

                tscalar gx = 0;
                const auto grad_time = nano::measure_robustly_nsec([&] ()
                {
                        problem(x, g);
                        gx += g.template lpNorm<Eigen::Infinity>();
                }, trials).count();

                auto& row = table.append(function.name());
                row << fval_time << grad_time;
        }
}

int main(int argc, char* argv[])
{
        using namespace nano;

        // parse the command line
        nano::cmdline_t cmdline("benchmark test functions");
        cmdline.add("", "min-dims",     "minimum number of dimensions for each test function (if feasible)", "128");
        cmdline.add("", "max-dims",     "maximum number of dimensions for each test function (if feasible)", "1024");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_dims = cmdline.get<tensor_size_t>("min-dims");
        const auto max_dims = cmdline.get<tensor_size_t>("max-dims");

        nano::table_t table("function");
        table.header() << "f(x) [ns]" << "f(x, g) [ns]";

        nano::foreach_test_function<scalar_t, nano::test_type::all>(min_dims, max_dims,
                [&] (const nano::function_t<scalar_t>& function)
        {
                eval_func(function, table);
        });

        table.print(std::cout);

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

