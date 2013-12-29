#include "ncv.h"
#include "models/forward_network.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();

        using namespace ncv;

        const string_t cmd_layer = "conv:count=16,rows=8,cols=8;";
        const string_t cmd_layer_snorm = cmd_layer + "snorm;";
        const string_t cmd_layer_tanh = cmd_layer + "tanh;";
        const string_t cmd_layer_unit = cmd_layer + "unit;";

        strings_t cmd_networks =
        {
                "",

                cmd_layer_snorm,
                cmd_layer_tanh,
                cmd_layer_unit,

                cmd_layer_snorm + cmd_layer_snorm,
                cmd_layer_tanh + cmd_layer_tanh,
                cmd_layer_unit + cmd_layer_unit,

                cmd_layer_snorm + cmd_layer_snorm + cmd_layer_snorm,
                cmd_layer_tanh + cmd_layer_tanh + cmd_layer_tanh,
                cmd_layer_unit + cmd_layer_unit + cmd_layer_unit
        };

        const color_mode cmd_color = color_mode::rgba;
        const size_t cmd_rows = 32;
        const size_t cmd_cols = 32;
        const size_t cmd_outputs = 10;
        const size_t cmd_samples = 10000;

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
                        count += output.size();
                }

                log_info() << "<<< processed [" << (count / cmd_outputs) << "] samples in " << timer.elapsed() << ".";
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
