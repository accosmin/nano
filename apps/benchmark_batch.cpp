#include "math/abs.hpp"
#include "min/batch.hpp"
#include "math/clamp.hpp"
#include "cortex/table.h"
#include "cortex/logger.h"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "cortex/optimizer.h"
#include "text/from_string.hpp"
#include "text/starts_with.hpp"
#include "min/func/run_all.hpp"
#include "benchmark_optimizers.h"
#include <map>
#include <tuple>

namespace
{
        using namespace cortex;

        template
        <
                typename tscalar,
                typename tostats,
                typename tsize = typename min::function_t<tscalar>::tsize,
                typename tvector = typename min::function_t<tscalar>::tvector,
                typename tproblem = typename min::function_t<tscalar>::tproblem
        >
        void check_function(const min::function_t<tscalar>& function, tostats& gstats)
        {
                const auto iterations = size_t(1024);
                const auto epsilon = math::epsilon0<tscalar>();
                const auto trials = size_t(1024);

                const size_t dims = function.problem().size();

                math::random_t<tscalar> rgen(-1.0, +1.0);

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
                        min::batch_optimizer::GD,
                        min::batch_optimizer::CGD_CD,
                        min::batch_optimizer::CGD_DY,
                        min::batch_optimizer::CGD_FR,
                        min::batch_optimizer::CGD_HS,
                        min::batch_optimizer::CGD_LS,
                        min::batch_optimizer::CGD_DYCD,
                        min::batch_optimizer::CGD_DYHS,
                        min::batch_optimizer::CGD_PRP,
                        min::batch_optimizer::CGD_N,
                        min::batch_optimizer::LBFGS
                };

                // line search initialization methods to try
                const auto ls_initializers =
                {
                        min::ls_initializer::unit,
                        min::ls_initializer::quadratic,
                        min::ls_initializer::consistent
                };

                // line search strategies to try
                const auto ls_strategies =
                {
                        min::ls_strategy::backtrack_armijo,
                        min::ls_strategy::backtrack_wolfe,
                        min::ls_strategy::backtrack_strong_wolfe,
                        min::ls_strategy::interpolation,
                        min::ls_strategy::cg_descent
                };

                // per-problem statistics
                tostats stats;

                // evaluate all possible combinations
                for (min::batch_optimizer optimizer : optimizers)
                        for (min::ls_initializer ls_init : ls_initializers)
                                for (min::ls_strategy ls_strat : ls_strategies)
                {
                        const auto op = [&] (const tproblem& problem, const tvector& x0)
                        {
                                return  min::minimize(
                                        problem, nullptr, x0, optimizer, iterations, epsilon, ls_init, ls_strat);
                        };

                        const string_t name =
                                text::to_string(optimizer) + "[" +
                                text::to_string(ls_init) + "][" +
                                text::to_string(ls_strat) + "]";

                        benchmark::benchmark_function(function, x0s, op, name, { 1e-12, 1e-10, 1e-8, 1e-6 }, stats, gstats);
                }

                // show per-problem statistics
                benchmark::show_table(function.name(), stats);
        }
}

int main(int, char* [])
{
        using namespace cortex;

        std::map<string_t, benchmark::optimizer_stat_t> gstats;

        min::run_all_test_functions<double>(8, [&] (const auto& function)
        {
                check_function(function, gstats);
        });

        // show global statistics
        benchmark::show_table(string_t(), gstats);

        // show per-optimizer statistics
        const auto optimizers =
        {
                min::batch_optimizer::GD,
                min::batch_optimizer::CGD_CD,
                min::batch_optimizer::CGD_DY,
                min::batch_optimizer::CGD_FR,
                min::batch_optimizer::CGD_HS,
                min::batch_optimizer::CGD_LS,
                min::batch_optimizer::CGD_DYCD,
                min::batch_optimizer::CGD_DYHS,
                min::batch_optimizer::CGD_PRP,
                min::batch_optimizer::CGD_N,
                min::batch_optimizer::LBFGS
        };

        for (min::batch_optimizer optimizer : optimizers)
        {
                const string_t name = text::to_string(optimizer) + "[";

                std::map<string_t, benchmark::optimizer_stat_t> stats;
                for (const auto& gstat : gstats)
                {
                        if (text::starts_with(gstat.first, name))
                        {
                                stats[gstat.first] = gstat.second;
                        }
                }

                benchmark::show_table(string_t(), stats);
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

