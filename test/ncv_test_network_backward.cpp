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

        string_t lmodel0;
        string_t lmodel1 = lmodel0 + "linear:dims=100;snorm;";
        string_t lmodel2 = lmodel1 + "linear:dims=100;snorm;";
        string_t lmodel3 = lmodel2 + "linear:dims=100;snorm;";

        string_t cmodel1;
        cmodel1 = cmodel1 + "conv:dims=32,rows=7,cols=7;snorm;smax-abs-pool;";
        cmodel1 = cmodel1 + "conv:dims=32,rows=4,cols=4;snorm;smax-abs-pool;";
        cmodel1 = cmodel1 + "conv:dims=32,rows=4,cols=4;snorm;";

        string_t cmodel2;
        cmodel2 = cmodel2 + "conv:dims=32,rows=7,cols=7;snorm;smax-abs-pool;";
        cmodel2 = cmodel2 + "conv:dims=32,rows=6,cols=6;snorm;smax-abs-pool;";
        cmodel2 = cmodel2 + "conv:dims=32,rows=3,cols=3;snorm;";

        string_t cmodel3;
        cmodel3 = cmodel3 + "conv:dims=32,rows=5,cols=5;snorm;smax-abs-pool;";
        cmodel3 = cmodel3 + "conv:dims=32,rows=5,cols=5;snorm;smax-abs-pool;";
        cmodel3 = cmodel3 + "conv:dims=32,rows=4,cols=4;snorm;";

        strings_t cmd_networks =
        {
                lmodel0,
                lmodel1,
                lmodel2,
                lmodel3,

                cmodel1,
                cmodel2,
                cmodel3
        };

        const logistic_loss_t loss;

        for (const string_t& cmd_network : cmd_networks)
        {
                log_info() << "<<< running network [" << cmd_network << "] ...";

                // create feed-forward network
                ncv::forward_network_t model(cmd_network);
                model.resize(cmd_rows, cmd_cols, cmd_outputs, cmd_color, true);

                // create random samples
                tensors_t samples(cmd_samples, tensor_t(cmd_color == color_mode::luma ? 1 : 3, cmd_rows, cmd_cols));

                random_t<scalar_t> rgen(-1.0, +1.0);
                for (tensor_t& sample : samples)
                {
                        sample.random(rgen);
                }

                // create random targets
                vectors_t targets(cmd_samples, vector_t(cmd_outputs));
                for (vector_t& target : targets)
                {
                        target.setRandom();
                }

                // process the samples
                ncv::timer_t timer;

                trainer_data_t gdata(model, trainer_data_t::type::vgrad);
                for (size_t i = 0; i < cmd_samples; i ++)
                {
                        gdata.update(samples[i], targets[i], loss);
                }

                log_info() << "<<< processed [" << gdata.count() << "] samples in " << timer.elapsed() << ".";
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
