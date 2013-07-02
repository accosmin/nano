#include "ncv.h"
#include "model/conv_layer.h"
#include "loss/loss_classnll.h"
#include <boost/program_options.hpp>

static const ncv::tensor3d_t& forward(ncv::conv_layers_t& layers, const ncv::tensor3d_t& _input)
{
        const ncv::tensor3d_t* input = &_input;
        for (ncv::conv_layers_t::const_iterator it = layers.begin(); it != layers.end(); ++ it)
        {
                input = &it->forward(*input);
        }

        return *input;
}

static void backward(ncv::conv_layers_t& layers, const ncv::tensor3d_t& _gradient)
{
        const ncv::tensor3d_t* gradient = &_gradient;
        for (ncv::conv_layers_t::reverse_iterator it = layers.rbegin(); it != layers.rend(); ++ it)
        {
                gradient = &it->backward(*gradient);
        }
}

int main(int argc, char *argv[])
{
        const ncv::size_t cmd_samples = 4;
        const ncv::size_t cmd_isize = 16;
        const ncv::size_t cmd_layers = 2;

        const ncv::size_t cmd_inputs = 1;
        const ncv::size_t cmd_outputs = 10;
        const ncv::size_t cmd_tests = 16;

        // generate random samples & targets
        ncv::tensor3ds_t samples(cmd_samples, ncv::tensor3d_t(1, cmd_isize, cmd_isize));
        for (ncv::tensor3d_t& sample : samples)
        {
                sample.random();
        }

        ncv::vectors_t targets(cmd_samples, ncv::vector_t(cmd_outputs));
        for (ncv::vector_t& target : targets)
        {
                ncv::random_t<ncv::scalar_t> rgen(-1.0, +1.0);
                rgen(target.data(), target.data() + target.size());
        }

        // build the convolution network
        std::vector<ncv::size_t> network_params;
        for (ncv::size_t l = 0; l < cmd_layers; l ++)
        {
                ncv::random_t<ncv::size_t> rgen(2, 6);
                network_params.push_back(rgen());
                network_params.push_back(rgen());
                network_params.push_back(rgen());
        }

        ncv::conv_layers_t layers;
        const ncv::size_t n_parameters = ncv::conv_layer_t::make_network(
                cmd_inputs, cmd_isize, cmd_isize, network_params, cmd_outputs, layers);

        ncv::log_info() << "number of parameters: " << n_parameters << ".";
        ncv::conv_layer_t::print_network(layers);

        ncv::classnll_loss_t loss;

        // optimization problem: size
        auto opt_fn_size = [&] ()
        {
                return n_parameters;
        };

        // optimization problem: function value
        auto opt_fn_fval = [&] (const ncv::vector_t& x)
        {
                ncv::deserializer_t(x) >> layers;

                ncv::scalar_t lvalue = 0.0;
                ncv::size_t lcount = 0;

                for (ncv::size_t i = 0; i < samples.size(); i ++)
                {
                        const ncv::tensor3d_t& input = samples[i];
                        const ncv::vector_t& target = targets[i];

                        // forward: network output
                        const ncv::tensor3d_t& output = forward(layers, input);

                        // loss value
                        ncv::vector_t voutput(output.size());
                        ncv::serializer_t(voutput) << output;

                        lvalue += loss.value(target, voutput);
                        lcount ++;
                }

                lvalue /= (lcount == 0) ? 1.0 : lcount;
                return lvalue;
        };

        // optimization problem: function value & gradient
        auto opt_fn_fval_grad = [&] (const ncv::vector_t& x, ncv::vector_t& gx)
        {
                ncv::deserializer_t(x) >> layers;

                ncv::scalar_t lvalue = 0.0;
                ncv::size_t lcount = 0;
                for (ncv::conv_layer_t& layer : layers)
                {
                        layer.zero_grad();
                }

                for (ncv::size_t i = 0; i < samples.size(); i ++)
                {
                        const ncv::tensor3d_t& input = samples[i];
                        const ncv::vector_t& target = targets[i];

                        // forward: network output
                        const ncv::tensor3d_t& output = forward(layers, input);

                        // loss value & gradient
                        ncv::vector_t voutput(output.size());
                        ncv::serializer_t(voutput) << output;

                        lvalue += loss.value(target, voutput);
                        lcount ++;

                        const ncv::vector_t vgradient = loss.vgrad(target, voutput);

                        // backward: network gradient
                        ncv::tensor3d_t gradient(output.size(), 1, 1);
                        ncv::deserializer_t(vgradient) >> gradient;

                        backward(layers, gradient);
                }

                gx.resize(n_parameters);

                ncv::serializer_t s(gx);
                for (ncv::conv_layer_t& layer : layers)
                {
                        s << layer.gdata();
                }

                gx /= (lcount == 0) ? 1.0 : lcount;
                lvalue /= (lcount == 0) ? 1.0 : lcount;

                return lvalue;
        };

        // construct optimization problem: analytic gradient and finite difference approximation
        const ncv::optimize::problem_t problem_gd(opt_fn_size, opt_fn_fval, opt_fn_fval_grad);
        const ncv::optimize::problem_t problem_ax(opt_fn_size, opt_fn_fval);

        // evaluate the analytical gradient vs. the finite difference approximation
        for (ncv::size_t t = 0; t < cmd_tests; t ++)
        {
                ncv::random_t<ncv::scalar_t> rgen(-1.0, +1.0);

                ncv::vector_t x(n_parameters);
                rgen(x.data(), x.data() + x.size());

                ncv::vector_t gx_gd, gx_ax;
                problem_gd.f(x, gx_gd);
                problem_ax.f(x, gx_ax);

                ncv::log_info() << "[" << (t + 1) << "/" << cmd_tests
                                << "]: gradient difference (analytic vs. finite difference) = "
                                << (gx_gd - gx_ax).lpNorm<Eigen::Infinity>() << ".";
        }

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
