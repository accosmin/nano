#include "math/abs.hpp"
#include "text/table.h"
#include "math/batch.hpp"
#include "math/clamp.hpp"
#include "text/cmdline.h"
#include "math/random.hpp"
#include "cortex/logger.h"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "text/algorithm.h"
#include "cortex/optimizer.h"
#include "text/from_string.hpp"
#include "math/funcs/foreach.hpp"
#include "benchmark_optimizers.h"
#include <map>
#include <tuple>

namespace
{
        using namespace nano;

        template
        <
                typename tscalar,
                typename tostats,
                typename tsize = typename nano::function_t<tscalar>::tsize,
                typename tvector = typename nano::function_t<tscalar>::tvector,
                typename tproblem = typename nano::function_t<tscalar>::tproblem
        >
        void check_function(const nano::function_t<tscalar>& function,
                const size_t trials, const size_t iterations,
                tostats& gstats)
        {
                const auto epsilon = nano::epsilon0<tscalar>();
                const auto dims = function.problem().size();

                nano::random_t<tscalar> rgen(tscalar(-1), tscalar(+1));

                // generate fixed random trials
                std::vector<tvector> x0s(trials);
                for (auto& x0 : x0s)
                {
                        x0.resize(dims);
                        rgen(x0.data(), x0.data() + x0.size());
                }

                // optimizers to try
                const auto optimizers =
                {
                        nano::batch_optimizer::GD,
                        nano::batch_optimizer::CGD_CD,
                        nano::batch_optimizer::CGD_DY,
                        nano::batch_optimizer::CGD_FR,
                        nano::batch_optimizer::CGD_HS,
                        nano::batch_optimizer::CGD_LS,
                        nano::batch_optimizer::CGD_DYCD,
                        nano::batch_optimizer::CGD_DYHS,
                        nano::batch_optimizer::CGD_PRP,
                        nano::batch_optimizer::CGD_N,
                        nano::batch_optimizer::LBFGS
                };

                // line search initialization methods to try
                const auto ls_initializers =
                {
                        nano::ls_initializer::unit,
                        nano::ls_initializer::quadratic,
                        nano::ls_initializer::consistent
                };

                // line search strategies to try
                const auto ls_strategies =
                {
                        nano::ls_strategy::backtrack_armijo,
                        nano::ls_strategy::backtrack_wolfe,
                        nano::ls_strategy::backtrack_strong_wolfe,
                        nano::ls_strategy::interpolation,
                        nano::ls_strategy::cg_descent
                };

                // per-problem statistics
                tostats stats;

                // evaluate all possible combinations
                for (nano::batch_optimizer optimizer : optimizers)
                        for (nano::ls_initializer ls_init : ls_initializers)
                                for (nano::ls_strategy ls_strat : ls_strategies)
                {
                        const auto op = [&] (const tproblem& problem, const tvector& x0)
                        {
                                return  nano::minimize(
                                        problem, nullptr, x0, optimizer, iterations, epsilon, ls_init, ls_strat);
                        };

                        const auto name =
                                nano::to_string(optimizer) + "[" +
                                nano::to_string(ls_init) + "][" +
                                nano::to_string(ls_strat) + "]";

                        benchmark::benchmark_function(function, x0s, op, name, { 1e-12, 1e-10, 1e-8, 1e-6 }, stats, gstats);
                }

                // show per-problem statistics
                benchmark::show_table(function.name(), stats);
        }
}

int main(int argc, char* argv[])
{
        using namespace nano;

        // parse the command line
        nano::cmdline_t cmdline("benchmark batch optimizers");
        cmdline.add("", "min-dims",     "minimum number of dimensions for each test function (if feasible)", "1");
        cmdline.add("", "max-dims",     "maximum number of dimensions for each test function (if feasible)", "8");
        cmdline.add("", "trials",       "number of random trials for each test function", "1024");
        cmdline.add("", "iterations",   "maximum number of iterations", "8000");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_dims = cmdline.get<tensor_size_t>("min-dims");
        const auto max_dims = cmdline.get<tensor_size_t>("max-dims");
        const auto trials = cmdline.get<size_t>("trials");
        const auto iterations = cmdline.get<size_t>("iterations");

        std::map<std::string, benchmark::optimizer_stat_t> gstats;

        nano::foreach_test_function<scalar_t, nano::test_type::all>(min_dims, max_dims,
                [&] (const nano::function_t<scalar_t>& function)
        {
                check_function(function, trials, iterations, gstats);
        });

        // show global statistics
        benchmark::show_table(std::string(), gstats);

        // show per-optimizer statistics
        const auto optimizers =
        {
                nano::batch_optimizer::GD,
                nano::batch_optimizer::CGD_CD,
                nano::batch_optimizer::CGD_DY,
                nano::batch_optimizer::CGD_FR,
                nano::batch_optimizer::CGD_HS,
                nano::batch_optimizer::CGD_LS,
                nano::batch_optimizer::CGD_DYCD,
                nano::batch_optimizer::CGD_DYHS,
                nano::batch_optimizer::CGD_PRP,
                nano::batch_optimizer::CGD_N,
                nano::batch_optimizer::LBFGS
        };

        for (nano::batch_optimizer optimizer : optimizers)
        {
                const auto name = nano::to_string(optimizer) + "[";

                std::map<std::string, benchmark::optimizer_stat_t> stats;
                for (const auto& gstat : gstats)
                {
                        if (nano::starts_with(gstat.first, name))
                        {
                                stats[gstat.first] = gstat.second;
                        }
                }

                benchmark::show_table(std::string(), stats);
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

