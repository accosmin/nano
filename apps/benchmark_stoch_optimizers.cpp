#include "core/table.h"
#include "math/abs.hpp"
#include "core/logger.h"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "core/minimize.h"
#include "math/numeric.hpp"
#include "text/from_string.hpp"
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
                const auto epochs = opt_size_t(128);
                const auto epoch_size = opt_size_t(32);
                const auto trials = size_t(1024);

                const auto dims = func.problem().size();

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
                        const auto op = [&] (const opt_problem_t& problem, const vector_t& x0)
                        {
                                opt_scalar_t alpha0, decay;
                                ncv::tune_stochastic(problem, x0, optimizer, epoch_size, alpha0, decay);

                                return  ncv::minimize(
                                        problem, nullptr, x0, optimizer, epochs, epoch_size, alpha0, decay);
                        };

                        const string_t name =
                                text::to_string(optimizer);

                        benchmark::benchmark_function(func, x0s, op, name, { 1e-5, 1e-4, 1e-3, 1e-2 }, stats, gstats);
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

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}

