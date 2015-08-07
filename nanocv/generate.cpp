#include "generate.h"
#include "model.h"
#include "minimize.h"
#include "nanocv/timer.h"
#include "nanocv/string.h"
#include "nanocv/logger.h"
#include "losses/loss_square.h"
#include "nanocv/math/random.hpp"
#include "nanocv/tensor/random.hpp"
#include <fstream>

namespace ncv
{
        tensor_t generate_match_target(const model_t& model, const vector_t& target)
        {
                const square_loss_t loss;

                // construct the optimization problem
                const timer_t timer;

                auto fn_size = [&] ()
                {
                        return model.isize();
                };

                auto fn_fval = [&] (const vector_t& x)
                {
                        const tensor_t output = model.output(x);

                        return loss.value(target, output.vector());
                };

                auto fn_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        const tensor_t output = model.output(x);
                        const vector_t ograd = loss.vgrad(target, output.vector());

                        gx = model.ginput(ograd).vector();

                        return loss.value(target, output.vector());
                };

                auto fn_wlog = [] (const string_t& message)
                {
                        log_warning() << message;
                };
                auto fn_elog = [] (const string_t& message)
                {
                        log_error() << message;
                };
                auto fn_ulog = [&] (const opt_state_t& /*result*/)
                {
//                        log_info() << "[loss = " << result.f
//                                   << ", grad = " << result.g.lpNorm<Eigen::Infinity>()
//                                   << ", funs = " << result.n_fval_calls() << "/" << result.n_grad_calls()
//                                   << "] done in " << timer.elapsed() << ".";

                        return true;
                };

                // assembly optimization problem & optimize the input
                const optim::batch_optimizer optimizer = optim::batch_optimizer::LBFGS;
                const size_t iterations = 256;
                const scalar_t epsilon = 1e-6;

                tensor_t input(model.idims(), model.irows(), model.icols());
                tensor::set_random(input, random_t<scalar_t>(0.0, 1.0));

                const opt_state_t result = ncv::minimize(
                        fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog,
                        input.vector(), optimizer, iterations, epsilon);

                input.vector() = result.x;

                log_info() << "[loss = " << result.f
                           << ", grad = " << result.g.lpNorm<Eigen::Infinity>()
                           << ", funs = " << result.n_fval_calls() << "/" << result.n_grad_calls()
                           << "] done in " << timer.elapsed() << ".";

                // OK
                return input;
        }
}
