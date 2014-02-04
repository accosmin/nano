#include "ncv.h"
#include "models/forward_network.h"
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

        string_t model0 = "";
        string_t model1 = "";
        string_t model2 = "";
        string_t model3 = "";

        model1 = model1 + "conv:count=32,rows=7,cols=7;snorm;smax-abs-pool;";
        model1 = model1 + "conv:count=32,rows=4,cols=4;snorm;smax-abs-pool;";
        model1 = model1 + "conv:count=32,rows=4,cols=4;snorm;";

        model2 = model2 + "conv:count=32,rows=7,cols=7;snorm;smax-abs-pool;";
        model2 = model2 + "conv:count=32,rows=6,cols=6;snorm;smax-abs-pool;";
        model2 = model2 + "conv:count=32,rows=3,cols=3;snorm;";

        model3 = model3 + "conv:count=32,rows=5,cols=5;snorm;smax-abs-pool;";
        model3 = model3 + "conv:count=32,rows=5,cols=5;snorm;smax-abs-pool;";
        model3 = model3 + "conv:count=32,rows=4,cols=4;snorm;";

        strings_t cmd_networks =
        {
                model0,
                model1,
                model2,
                model3
        };

        for (const string_t& cmd_network : cmd_networks)
        {
                log_info() << "<<< running network [" << cmd_network << "] ...";

                // create feed-forward network
                ncv::forward_network_t network(cmd_network);
                network.resize(cmd_rows, cmd_cols, cmd_outputs, cmd_color);

                // create random samples
                tensor3ds_t samples(cmd_samples, tensor3d_t(cmd_color == color_mode::luma ? 1 : 3, cmd_rows, cmd_cols));
                for (tensor3d_t& sample : samples)
                {
                        sample.random(-1.0, +1.0);
                }

                // process the samples
                ncv::timer_t timer;

                size_t count = 0;
                for (tensor3d_t& sample : samples)
                {
                        const vector_t output = network.value(sample);
			const vector_t gradient = network.gradient(output);
                        count += output.size() + gradient.size();
                }

                log_info() << "<<< processed ["
                           << ((count - network.n_parameters() * samples.size()) / cmd_outputs)
                           << "] samples in " << timer.elapsed() << ".";
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
