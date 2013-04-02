#include "ncv.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();
        
        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "help message");
        po_desc.add_options()("input,i",
                boost::program_options::value<ncv::string_t>(),
                "input image path");
        po_desc.add_options()("channel,c",
                boost::program_options::value<ncv::string_t>()->default_value("luma"),
                ("color channel (" +
                ncv::text::to_string(ncv::channel::red) + ", " +
                ncv::text::to_string(ncv::channel::green) + ", " +
                ncv::text::to_string(ncv::channel::blue) + ", " +
                ncv::text::to_string(ncv::channel::luma) + ")").c_str());
        po_desc.add_options()("scale,s",
                boost::program_options::value<ncv::scalar_t>()->default_value(1.0),
                "scaling factor [0.1, 10.0]");
        po_desc.add_options()("width,w",
                boost::program_options::value<ncv::index_t>()->default_value(0),
                "scaling width [0, 4096] (considered if positive)");
        po_desc.add_options()("height,h",
                boost::program_options::value<ncv::index_t>()->default_value(0),
                "scaling height [0, 4096] (considered if positive)");
        po_desc.add_options()("output,o",
                boost::program_options::value<ncv::string_t>(),
                "output image path");
	
        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);
        		
        // check arguments and options
        if (	po_vm.empty() ||
                !po_vm.count("input") ||
                !po_vm.count("output") ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const ncv::string_t cmd_input = po_vm["input"].as<ncv::string_t>();
        const ncv::channel cmd_channel = ncv::text::from_string<ncv::channel>(po_vm["channel"].as<ncv::string_t>());
        const ncv::scalar_t cmd_scale = ncv::math::clamp(po_vm["scale"].as<ncv::scalar_t>(), 0.1, 10.0);
        const ncv::index_t cmd_width = ncv::math::clamp(po_vm["width"].as<ncv::index_t>(), 0, 4096);
        const ncv::index_t cmd_height = ncv::math::clamp(po_vm["height"].as<ncv::index_t>(), 0, 4096);
        const ncv::string_t cmd_output = po_vm["output"].as<ncv::string_t>();

        ncv::timer timer;

        // load input image
        ncv::image image;
        ncv::pixel_matrix_t idata, odata;

        timer.start();
        if (    image.load(cmd_input) == false ||
                image.save<ncv::pixel_t>(idata, cmd_channel) == false)
        {
                ncv::log_error() << "<<< failed to load image <" << cmd_input << ">!";
                return EXIT_FAILURE;
        }
        else
        {
                ncv::log_info() << "<<< loaded image <" << cmd_input << "> in " << timer.elapsed_string() << ".";
        }

        // scale image
        timer.start();
        if (cmd_width > 0 && cmd_height > 0)
        {
                ncv::math::scale(idata, odata, cmd_width, cmd_height);
        }
        else
        {
                ncv::math::scale(idata, odata, cmd_scale);
        }
        ncv::log_info() << "scaled image from <" << idata.cols() << "x" << idata.rows()
                        << "> to <" << odata.cols() << "x" << odata.rows()
                        << "> in " << timer.elapsed_string() << ".";

        // save output image
        timer.start();
        if (    image.load<ncv::pixel_t>(odata, cmd_channel) == false ||
                image.save(cmd_output) == false)
        {
                ncv::log_error() << ">>> failed to save image <" << cmd_output << ">!";
                return EXIT_FAILURE;
        }
        else
        {
                ncv::log_info() << ">>> saved image <" << cmd_output << "> in " << timer.elapsed_string() << ".";
        }
		
        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
