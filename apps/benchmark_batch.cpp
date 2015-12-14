#include "math/abs.hpp"
#include "text/table.h"
#include "math/batch.hpp"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "text/algorithm.h"
#include "cortex/optimizer.h"
#include "cortex/util/logger.h"
#include "text/from_string.hpp"
#include "math/funcs/foreach.hpp"
#include "benchmark_optimizers.h"
#include <map>
#include <tuple>
#include <boost/program_options.hpp>

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
                const size_t trials, const size_t iterations,
                tostats& gstats)
        {
                const auto epsilon = math::epsilon0<tscalar>();
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

                        const auto name =
                                text::to_string(optimizer) + "[" +
                                text::to_string(ls_init) + "][" +
                                text::to_string(ls_strat) + "]";

                        benchmark::benchmark_function(function, x0s, op, name, { 1e-12, 1e-10, 1e-8, 1e-6 }, stats, gstats);
                }

                // show per-problem statistics
                benchmark::show_table(function.name(), stats);
        }
}

int main(int argc, char* argv[])
{
        using namespace cortex;

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "benchmark batch optimizers");
        po_desc.add_options()("min-dims",
                boost::program_options::value<tensor_size_t>()->default_value(1),
                "minimum number of dimensions for each test function (if feasible)");
        po_desc.add_options()("max-dims",
                boost::program_options::value<tensor_size_t>()->default_value(8),
                "maximum number of dimensions for each test function (if feasible)");
        po_desc.add_options()("trials",
                boost::program_options::value<size_t>()->default_value(1024),
                "number of random trials for each test function");
        po_desc.add_options()("iterations",
                boost::program_options::value<size_t>()->default_value(8000),
                "maximum number of iterations");

        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);

        // check arguments and options
        if (	po_vm.empty() ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const auto min_dims = po_vm["min-dims"].as<tensor_size_t>();
        const auto max_dims = po_vm["max-dims"].as<tensor_size_t>();
        const auto trials = po_vm["trials"].as<size_t>();
        const auto iterations = po_vm["iterations"].as<size_t>();

        std::map<std::string, benchmark::optimizer_stat_t> gstats;

        math::foreach_test_function<scalar_t, math::test_type::all>(min_dims, max_dims, [&] (const auto& function)
        {
                check_function(function, trials, iterations, gstats);
        });

        // show global statistics
        benchmark::show_table(std::string(), gstats);

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
                const auto name = text::to_string(optimizer) + "[";

                std::map<std::string, benchmark::optimizer_stat_t> stats;
                for (const auto& gstat : gstats)
                {
                        if (text::starts_with(gstat.first, name))
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

