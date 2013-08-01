#include "ncv.h"
#include "core/logger.h"
#include "core/timer.h"
#include "core/clamp.h"
#include "core/bilinear.h"
#include "core/transform.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();
        
        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "help message");
        po_desc.add_options()("input",
                boost::program_options::value<ncv::string_t>(),
                "input image path");
        po_desc.add_options()("scale",
                boost::program_options::value<ncv::scalar_t>()->default_value(1.0),
                "scaling factor [0.1, 10.0]");
        po_desc.add_options()("width",
                boost::program_options::value<ncv::size_t>()->default_value(0),
                "scaling width [0, 4096] (considered if positive)");
        po_desc.add_options()("height",
                boost::program_options::value<ncv::size_t>()->default_value(0),
                "scaling height [0, 4096] (considered if positive)");
        po_desc.add_options()("output",
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
        const ncv::scalar_t cmd_scale = ncv::math::clamp(po_vm["scale"].as<ncv::scalar_t>(), 0.1, 10.0);
        const ncv::size_t cmd_width = ncv::math::clamp(po_vm["width"].as<ncv::size_t>(), 0, 4096);
        const ncv::size_t cmd_height = ncv::math::clamp(po_vm["height"].as<ncv::size_t>(), 0, 4096);
        const ncv::string_t cmd_output = po_vm["output"].as<ncv::string_t>();

        ncv::timer_t timer;

        // load input image
        ncv::rgba_matrix_t rgba_image;
        ncv::cielab_matrix_t cielab_image, cielab_image_scaled;

        timer.start();
        if (!ncv::load_rgba(cmd_input, rgba_image))
        {
                ncv::log_error() << "<<< failed to load image <" << cmd_input << ">!";
                return EXIT_FAILURE;
        }
        else
        {
                ncv::log_info() << "<<< loaded image <" << cmd_input << "> in " << timer.elapsed() << ".";
        }

        // transform RGBA to CIELab
        timer.start();
        ncv::math::transform(rgba_image, cielab_image, ncv::color::make_cielab);
        ncv::log_info() << "transformed RGBA to CIELab in " << timer.elapsed() << ".";

        // resize image
        timer.start();
        if (cmd_width > 0 && cmd_height > 0)
        {
                ncv::math::bilinear(cielab_image, cielab_image_scaled, cmd_width, cmd_height);
        }
        else
        {
                ncv::math::bilinear(cielab_image, cielab_image_scaled, cmd_scale);
        }
        ncv::log_info() << "scaled image from <" << cielab_image.cols() << "x" << cielab_image.rows()
                        << "> to <" << cielab_image_scaled.cols() << "x" << cielab_image_scaled.rows()
                        << "> in " << timer.elapsed() << ".";

        // transform CIELab to RGBA
        timer.start();
        ncv::math::transform(cielab_image_scaled, rgba_image,
                             [](const ncv::cielab_t& cielab) { return ncv::color::make_rgba(cielab); });
        ncv::log_info() << "transformed CIELab to RGBA in " << timer.elapsed() << ".";

        // save output image
        timer.start();
        if (!ncv::save_rgba(cmd_output, rgba_image))
        {
                ncv::log_error() << ">>> failed to save image <" << cmd_output << ">!";
                return EXIT_FAILURE;
        }
        else
        {
                ncv::log_info() << ">>> saved image <" << cmd_output << "> in " << timer.elapsed() << ".";
        }
		
        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
