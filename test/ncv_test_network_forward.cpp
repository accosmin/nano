#include "ncv.h"
#include "models/forward_network.h"
#include "losses/loss_logistic.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();

        using namespace ncv;

        const color_mode cmd_color = color_mode::luma;
        const size_t cmd_rows = 28;
        const size_t cmd_cols = 28;
        const size_t cmd_outputs = 10;
        const size_t cmd_samples = 10000;

        strings_t cmd_networks =
        {
                "",

                "linear:dims=10",
                "linear:dims=100",
                "linear:dims=1000",

                "linear:dims=100;linear:dims=100",
                "linear:dims=100;linear:dims=100;linear:dims=100",
                "linear:dims=100;linear:dims=100;linear:dims=100;linear:dims=100"
        };

        const logistic_loss_t loss;

        for (const string_t& cmd_network : cmd_networks)
        {
                log_info() << "<<< running network [" << cmd_network << "] ...";

                // create feed-forward network
                forward_network_t model(cmd_network);
                model.resize(cmd_rows, cmd_cols, cmd_outputs, cmd_color);

                // create random samples
                tensors_t samples(cmd_samples, tensor_t(cmd_color == color_mode::luma ? 1 : 3, 1, cmd_rows, cmd_cols));
                for (tensor_t& sample : samples)
                {
                        sample.random(random_t<scalar_t>(-1.0, +1.0));
                }

                // create random targets
                vectors_t targets(cmd_samples, vector_t(cmd_outputs));
                for (vector_t& target : targets)
                {
                        target.setRandom();
                }

                // process the samples
                ncv::timer_t timer;

                trainer_data_t ldata(model, trainer_data_t::type::value);
                for (size_t i = 0; i < cmd_samples; i ++)
                {
                        ldata.update(samples[i], targets[i], loss);
                }

                log_info() << "<<< processed [" << ldata.count() << "] samples in " << timer.elapsed() << ".";
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
