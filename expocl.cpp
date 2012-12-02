#define ARRAY_SIZE 64

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <memory>

#include <vigra/impex.hxx>
#include <vigra/stdimage.hxx>
#include <vigra/transformimage.hxx>

#define __CL_ENABLE_EXCEPTIONS
#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

/* Find a GPU or CPU associated with the first available platform */
void create_devices(std::vector<cl::Device>& devices)
{
    std::vector<cl::Platform> platforms;

    /* Identify a platform */
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    /* Gather devices */
    try {
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    }
    catch (cl::Error& e) {
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
    }
}

std::string slurp(std::ifstream const& in) {
    if (!in.good()) {
        throw std::runtime_error("Could not read from input file stream.");
    }
    std::stringstream sstr;
    sstr << in.rdbuf();
    return sstr.str();
}

/* Create program from a file and compile it */
cl::Program build_program(cl::Context const& ctx, cl::Device dev, char const* filename) {

    /* Read program file and place content into buffer */
    std::string program_source = slurp(std::ifstream(filename));

    /* Create program from file */
    cl::Program program(ctx, program_source);

    /* Build program */
    try {
        program.build();
    }
    catch (cl::Error const& e) {
        /* Output build log on failure */
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev) << std::endl;
        exit(1);
    }

    return program;
}

void process(vigra::FRGBImage const& image)
{


}

std::string basename(std::string const& path)
{
    std::cout << path.rfind('.') << std::endl;
    return path.substr(0, path.rfind('.'));
}

std::shared_ptr< vigra::BasicImage< vigra::RGBValue< vigra::UInt8 >>>
loadImage(char const* path)
{
    typedef vigra::BasicImage< vigra::RGBValue< vigra::UInt8 >> ImgType;
    vigra::ImageImportInfo info(path);

    if(info.isGrayscale())
    {
        throw std::runtime_error( "Could not open grayscale image. Only RGB images supported." );
    }

    // TODO: check pixel type is uint8

    auto img = std::make_shared<ImgType>( info.width(), info.height() );

    // import the image just read
    importImage(info, destImage(*img));

    return img;
}

struct OpenCLGPU
{
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;

    OpenCLGPU()
    {
        std::vector<cl::Device> devices;
        create_devices(devices);
        device = devices[0];

        context = cl::Context(device);

        queue = cl::CommandQueue(context, device);
    }

};

template <typename InComponentType>
void
saveTiff16(vigra::BasicImage< vigra::RGBValue< InComponentType >> const& in, std::string const& outPath)
{
    typedef vigra::RGBValue< InComponentType > InPixelType;
    typedef vigra::UInt16 OutComponentType;
    typedef vigra::RGBValue< OutComponentType > OutPixelType;
    typedef vigra::BasicImage< OutPixelType > OutImgType;

    vigra::ImageExportInfo exportInfo(outPath.c_str());
    exportInfo.setFileType("TIFF");
    exportInfo.setPixelType("UINT16");
    exportInfo.setCompression("LZW");

    OutImgType out(in.width(), in.height());

    // transform
    vigra::transformImage(in.upperLeft(), in.lowerRight(), in.accessor(),
            out.upperLeft(), out.accessor(),
            vigra::linearRangeMapping(
                InPixelType(vigra::NumericTraits<InComponentType>::min()),
                InPixelType(vigra::NumericTraits<InComponentType>::max()),
                OutPixelType(vigra::NumericTraits<OutComponentType>::min()),
                OutPixelType(vigra::NumericTraits<OutComponentType>::max())
                ));

    // write the image to the file given as second argument
    // the file type will be determined from the file name's extension
    exportImage(srcImageRange(out), exportInfo);
}

int main(int argc, char const *argv[])
{
    // create device, context, and queue
    OpenCLGPU gpu;

    /* Build program */
    cl::Program program = build_program(gpu.context, gpu.device, "expocl.cl");

    // for each image on the commandline, pass through openCL
    for (int i = 1; i < argc; ++i) {
        std::cout << argv[i] << std::endl;
        auto in = loadImage(argv[i]);

        // modify with OpenCL
        
        std::string outPath(basename(argv[i]));
        outPath += ".tiff";

        saveTiff16(*in, outPath);
    }

    exit(0);

    /* Data and buffers */
    float data[ARRAY_SIZE];
    float sum[2], total, actual_sum;
    cl_int num_groups;

    /* Initialize data */
    for(cl_int i = 0; i < ARRAY_SIZE; i++) {
        data[i] = 1.0f*i;
    }

    /* Create data buffer */
    size_t global_size = 8;
    size_t local_size = 4;
    num_groups = global_size/local_size;

    cl::Buffer input_buffer(gpu.context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            ARRAY_SIZE * sizeof(float),
            data);
    cl::Buffer sum_buffer(gpu.context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            num_groups * sizeof(float),
            sum);

    /* Create a kernel */
    cl::Kernel kernel(program, "add_numbers");

    /* Create kernel arguments */
    kernel.setArg(0, input_buffer);
    kernel.setArg(1, cl::Local(local_size * sizeof(float)));
    kernel.setArg(2, sum_buffer);

    /* Enqueue kernel */
    gpu.queue.enqueueNDRangeKernel(kernel,
                               cl::NullRange,
                               cl::NDRange(global_size),
                               cl::NDRange(local_size));

    /* Read the kernel's output */
    gpu.queue.enqueueReadBuffer(sum_buffer, CL_TRUE, 0, sizeof(sum), sum);

    /* Check result */
    total = 0.0f;
    for(cl_int j = 0; j < num_groups; j++) {
        total += sum[j];
    }
    actual_sum = 1.0f * ARRAY_SIZE/2*(ARRAY_SIZE-1);
    printf("Computed sum = %.1f.\n", total);
    if(fabs(total - actual_sum) > 0.01*fabs(actual_sum))
        printf("Check failed.\n");
    else
        printf("Check passed.\n");

    return 0;
}

