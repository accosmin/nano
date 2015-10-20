#include "math/abs.hpp"
#include "math/clamp.hpp"
#include "cortex/table.h"
#include "cortex/logger.h"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "min/tune_stoch.hpp"
#include "cortex/optimizer.h"
#include "text/from_string.hpp"
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
                const auto epochs = size_t(128);
                const auto epoch_size = size_t(32);
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
                        min::stoch_optimizer::SG,
                        min::stoch_optimizer::SGA,
                        min::stoch_optimizer::SIA,
                        min::stoch_optimizer::AG,
                        min::stoch_optimizer::AGGR,
                        min::stoch_optimizer::ADAGRAD,
                        min::stoch_optimizer::ADADELTA
                };

                // per-problem statistics
                tostats stats;

                // evaluate all optimizers
                for (const auto optimizer : optimizers)
                {
                        const auto op = [&] (const tproblem& problem, const tvector& x0)
                        {
                                tscalar alpha, decay;
                                min::tune_stochastic(
                                        problem, x0, optimizer, epoch_size, alpha, decay);

                                return  min::minimize(
                                        problem, nullptr, x0, optimizer, epochs, epoch_size, alpha, decay);
                        };

                        const string_t name =
                                text::to_string(optimizer);

                        benchmark::benchmark_function(function, x0s, op, name, { 1e-5, 1e-4, 1e-3, 1e-2 }, stats, gstats);
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

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

