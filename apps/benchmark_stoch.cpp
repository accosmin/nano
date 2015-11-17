#include "math/abs.hpp"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/tune_stoch.hpp"
#include "cortex/optimizer.h"
#include "cortex/util/table.h"
#include "cortex/util/logger.h"
#include "text/from_string.hpp"
#include "math/funcs/make_all.hpp"
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
                        math::stoch_optimizer::AG,
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
                                tscalar alpha, decay;
                                math::tune_stochastic(
                                        problem, x0, optimizer, epoch_size, alpha, decay);

                                return  math::minimize(
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

int main(int argc, char* argv[])
{
        using namespace cortex;

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "benchmark stochastic optimizers");
        po_desc.add_options()("min-dims",
                boost::program_options::value<tensor_size_t>()->default_value(1),
                "minimum number of dimensions for each test function (if feasible)");
        po_desc.add_options()("max-dims",
                boost::program_options::value<tensor_size_t>()->default_value(8),
                "maximum number of dimensions for each test function (if feasible)");
        po_desc.add_options()("trials",
                boost::program_options::value<size_t>()->default_value(1024),
                "number of random trials for each test function");
        po_desc.add_options()("epochs",
                boost::program_options::value<size_t>()->default_value(128),
                "number of epochs");
        po_desc.add_options()("epoch-size",
                boost::program_options::value<size_t>()->default_value(32),
                "number of iterations per epoch");

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
        const auto epochs = po_vm["epochs"].as<size_t>();
        const auto epoch_size = po_vm["epoch-size"].as<size_t>();

        std::map<string_t, benchmark::optimizer_stat_t> gstats;

        const auto functions = math::make_all_test_functions<scalar_t>(min_dims, max_dims);
        for (const auto& function : functions)
        {
                check_function(*function, trials, epochs, epoch_size, gstats);
        }

        // show global statistics
        benchmark::show_table(string_t(), gstats);

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

