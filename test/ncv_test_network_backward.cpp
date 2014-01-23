#include "ncv.h"
#include "models/forward_network.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();

        using namespace ncv;

        const string_t network0 = "";
        const string_t network1 = network0 + "conv:count=16,rows=9,cols=9;snorm;";
        const string_t network2 = network1 + "conv:count=16,rows=9,cols=9;snorm;";
        const string_t network3 = network2 + "conv:count=16,rows=7,cols=7;snorm;";
        const string_t network4 = network3 + "conv:count=16,rows=6,cols=6;snorm;";

        strings_t cmd_networks =
        {
                network0,
                network1,
                network2,
                network3,
                network4
        };

        const color_mode cmd_color = color_mode::luma;
        const size_t cmd_rows = 28;
        const size_t cmd_cols = 28;
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
