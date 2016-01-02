#include "math/abs.hpp"
#include "text/table.h"
#include "text/cmdline.h"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/tune_stoch.hpp"
#include "cortex/optimizer.h"
#include "cortex/util/logger.h"
#include "text/from_string.hpp"
#include "math/funcs/foreach.hpp"
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
        void check_function(const math::function_t<tscalar>& function,
                const size_t trials, const size_t epochs, const size_t epoch_size,
                tostats& gstats)
        {
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
                        math::stoch_optimizer::SG,
                        math::stoch_optimizer::SGA,
                        math::stoch_optimizer::SIA,
                        math::stoch_optimizer::SGM,
                        math::stoch_optimizer::AG,
                        math::stoch_optimizer::AGFR,
                        math::stoch_optimizer::AGGR,
                        math::stoch_optimizer::ADAGRAD,
                        math::stoch_optimizer::ADADELTA
                };

                // per-problem statistics
                tostats stats;

                // evaluate all optimizers
                for (const auto optimizer : optimizers)
                {
                        const auto op = [&] (const tproblem& problem, const tvector& x0)
                        {
                                tscalar alpha, decay, momentum;
                                math::tune_stochastic(
                                        problem, x0, optimizer, epoch_size, alpha, decay, momentum);

                                return  math::minimize(
                                        problem, nullptr, x0, optimizer, epochs, epoch_size, alpha, decay, momentum);
                        };

                        const auto name =
                                text::to_string(optimizer);

                        benchmark::benchmark_function(function, x0s, op, name, { 1e-5, 1e-4, 1e-3, 1e-2 }, stats, gstats);
                }

                // show per-problem statistics
                benchmark::show_table(function.name(), stats);
        }
}

int main(int argc, char* argv[])
{
        using namespace cortex;

        // parse the command line
        text::cmdline_t cmdline("benchmark stochastic optimizers");
        cmdline.add("", "min-dims",     "minimum number of dimensions for each test function (if feasible)", "1");
        cmdline.add("", "max-dims",     "maximum number of dimensions for each test function (if feasible)", "8");
        cmdline.add("" , "trials",      "number of random trials for each test function", "1024");
        cmdline.add("", "epochs",       "optimization: number of epochs", "128");
        cmdline.add("", "epoch-size",   "optimization: number of iterations per epoch", "32");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_dims = cmdline.get<tensor_size_t>("min-dims");
        const auto max_dims = cmdline.get<tensor_size_t>("max-dims");
        const auto trials = cmdline.get<size_t>("trials");
        const auto epochs = cmdline.get<size_t>("epochs");
        const auto epoch_size = cmdline.get<size_t>("epoch-size");

        std::map<std::string, benchmark::optimizer_stat_t> gstats;

        math::foreach_test_function<scalar_t, math::test_type::all>(min_dims, max_dims,
                [&] (const math::function_t<scalar_t>& function)
        {
                check_function(function, trials, epochs, epoch_size, gstats);
        });

        // show global statistics
        benchmark::show_table(std::string(), gstats);

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

