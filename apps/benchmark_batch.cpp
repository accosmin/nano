#include "math/abs.hpp"
#include "math/batch.hpp"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "cortex/optimizer.h"
#include "cortex/util/table.h"
#include "cortex/util/logger.h"
#include "text/from_string.hpp"
#include "text/starts_with.hpp"
#include "math/funcs/run_all.hpp"
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
                typename tsize = typename math::function_t<tscalar>::tsize,
                typename tvector = typename math::function_t<tscalar>::tvector,
                typename tproblem = typename math::function_t<tscalar>::tproblem
        >
        void check_function(const math::function_t<tscalar>& function, tostats& gstats)
        {
                const auto iterations = size_t(8 * 1024);
                const auto epsilon = math::epsilon0<tscalar>();
                const auto trials = size_t(1024);

                const auto dims = function.problem().size();

                math::random_t<tscalar> rgen(tscalar(-1), tscalar(+1));

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
                        math::batch_optimizer::GD,
                        math::batch_optimizer::CGD_CD,
                        math::batch_optimizer::CGD_DY,
                        math::batch_optimizer::CGD_FR,
                        math::batch_optimizer::CGD_HS,
                        math::batch_optimizer::CGD_LS,
                        math::batch_optimizer::CGD_DYCD,
                        math::batch_optimizer::CGD_DYHS,
                        math::batch_optimizer::CGD_PRP,
                        math::batch_optimizer::CGD_N,
                        math::batch_optimizer::LBFGS
                };

                // line search initialization methods to try
                const auto ls_initializers =
                {
                        math::ls_initializer::unit,
                        math::ls_initializer::quadratic,
                        math::ls_initializer::consistent
                };

                // line search strategies to try
                const auto ls_strategies =
                {
                        math::ls_strategy::backtrack_armijo,
                        math::ls_strategy::backtrack_wolfe,
                        math::ls_strategy::backtrack_strong_wolfe,
                        math::ls_strategy::interpolation,
                        math::ls_strategy::cg_descent
                };

                // per-problem statistics
                tostats stats;

                // evaluate all possible combinations
                for (math::batch_optimizer optimizer : optimizers)
                        for (math::ls_initializer ls_init : ls_initializers)
                                for (math::ls_strategy ls_strat : ls_strategies)
                {
                        const auto op = [&] (const tproblem& problem, const tvector& x0)
                        {
                                return  math::minimize(
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

        math::run_all_test_functions<double>(8, [&] (const auto& function)
        {
                check_function(function, gstats);
        });

        // show global statistics
        benchmark::show_table(string_t(), gstats);

        // show per-optimizer statistics
        const auto optimizers =
        {
                math::batch_optimizer::GD,
                math::batch_optimizer::CGD_CD,
                math::batch_optimizer::CGD_DY,
                math::batch_optimizer::CGD_FR,
                math::batch_optimizer::CGD_HS,
                math::batch_optimizer::CGD_LS,
                math::batch_optimizer::CGD_DYCD,
                math::batch_optimizer::CGD_DYHS,
                math::batch_optimizer::CGD_PRP,
                math::batch_optimizer::CGD_N,
                math::batch_optimizer::LBFGS
        };

        for (math::batch_optimizer optimizer : optimizers)
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

