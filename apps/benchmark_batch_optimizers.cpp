#include "core/table.h"
#include "math/abs.hpp"
#include "core/logger.h"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "core/minimize.h"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "text/from_string.hpp"
#include "text/starts_with.hpp"
#include "func/make_functions.h"
#include "benchmark_optimizers.h"
#include <map>
#include <tuple>

namespace
{
        using namespace ncv;

        template <typename tostats>
        void check_function(const function_t& func, tostats& gstats)
        {
                const auto iterations = opt_size_t(1024);
                const auto epsilon = math::epsilon0<opt_scalar_t>();
                const auto trials = size_t(1024);

                const size_t dims = func.problem().size();

                math::random_t<opt_scalar_t> rgen(-1.0, +1.0);

                // generate fixed random trials
                std::vector<opt_vector_t> x0s(trials);
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
                        const auto op = [&] (const opt_problem_t& problem, const vector_t& x0)
                        {
                                return  ncv::minimize(
                                        problem, nullptr, x0, optimizer, iterations, epsilon, ls_init, ls_strat);
                        };

                        const string_t name =
                                text::to_string(optimizer) + "[" +
                                text::to_string(ls_init) + "][" +
                                text::to_string(ls_strat) + "]";

                        benchmark::benchmark_function(func, x0s, op, name, { 1e-6, 1e-8, 1e-10, 1e-12 }, stats, gstats);
                }

                // show per-problem statistics
                benchmark::show_table(func.name(), stats);
        }
}

int main(int, char* [])
{
        using namespace ncv;

        std::map<string_t, benchmark::optimizer_stat_t> gstats;

        const auto funcs = ncv::make_all_test_functions(8);
        for (const auto& func : funcs)
        {
                check_function(*func, gstats);
        }

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

