#include "tensor.h"
#include "math/stats.hpp"
#include "text/table.h"
#include "text/cmdline.h"
#include "cortex/util/timer.h"
#include "cortex/util/logger.h"
#include "math/funcs/foreach.hpp"
#include "cortex/util/measure.hpp"
#include <iostream>

namespace
{
        using namespace zob;

        template
        <
                typename tscalar,
                typename tsize = typename zob::function_t<tscalar>::tsize,
                typename tvector = typename zob::function_t<tscalar>::tvector,
                typename tproblem = typename zob::function_t<tscalar>::tproblem
        >
        void eval_func(const zob::function_t<tscalar>& function, zob::table_t& table)
        {
                const auto problem = function.problem();

                zob::stats_t<scalar_t> fval_times;
                zob::stats_t<scalar_t> grad_times;

                const auto dims = function.problem().size();
                const tvector x = tvector::Zero(dims);
                tvector g = tvector::Zero(dims);

                const size_t trials = 16;

                tscalar fx = 0;
                const auto fval_time = zob::measure_robustly_nsec([&] ()
                {
                        fx += problem(x);
                }, trials).count();

                tscalar gx = 0;
                const auto grad_time = zob::measure_robustly_nsec([&] ()
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
        using namespace zob;

        // parse the command line
        zob::cmdline_t cmdline("benchmark test functions");
        cmdline.add("", "min-dims",     "minimum number of dimensions for each test function (if feasible)", "128");
        cmdline.add("", "max-dims",     "maximum number of dimensions for each test function (if feasible)", "1024");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_dims = cmdline.get<tensor_size_t>("min-dims");
        const auto max_dims = cmdline.get<tensor_size_t>("max-dims");

        zob::table_t table("function");
        table.header() << "f(x) [ns]" << "f(x, g) [ns]";

        zob::foreach_test_function<scalar_t, zob::test_type::all>(min_dims, max_dims,
                [&] (const zob::function_t<scalar_t>& function)
        {
                eval_func(function, table);
        });

        table.print(std::cout);

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

