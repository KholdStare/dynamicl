#include "cl_common.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace
{

    /* Find a GPU or CPU associated with the first available platform */
    void createDevices(std::vector<cl::Device>& devices)
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

    // TODO: actually get best device
    cl::Device getBestDevice()
    {
        std::vector<cl::Device> devices;
        createDevices(devices);

        return devices[0];
    }

    /**
     * Return a string containing the entire contents of a file
     */
    std::string slurp(std::ifstream const& in)
    {
        if (!in.good()) {
            throw std::runtime_error("Could not read from input file stream.");
        }
        std::stringstream sstr;
        sstr << in.rdbuf();
        return sstr.str();
    }

}

namespace DynamiCL
{

    char const* clErrorToStr(cl_int err)
    {
        switch (err)
        {
            case  0:    return  "SUCCESS";
            case  -1:   return  "DEVICE NOT FOUND";
            case  -2:   return  "DEVICE NOT AVAILABLE";
            case  -3:   return  "COMPILER NOT AVAILABLE";
            case  -4:   return  "MEM OBJECT ALLOCATION FAILURE";
            case  -5:   return  "OUT OF RESOURCES";
            case  -6:   return  "OUT OF HOST MEMORY";
            case  -7:   return  "PROFILING INFO NOT AVAILABLE";
            case  -8:   return  "MEM COPY OVERLAP";
            case  -9:   return  "IMAGE FORMAT MISMATCH";
            case  -10:  return  "IMAGE FORMAT NOT SUPPORTED";
            case  -11:  return  "BUILD PROGRAM FAILURE";
            case  -12:  return  "MAP FAILURE";
            case  -13:  return  "MISALIGNED SUB BUFFER OFFSET";
            case  -14:  return  "EXEC STATUS ERROR FOR EVENTS IN WAIT LIST";
            case  -15:  return  "COMPILE PROGRAM FAILURE";
            case  -16:  return  "LINKER NOT AVAILABLE";
            case  -17:  return  "LINK PROGRAM FAILURE";
            case  -18:  return  "DEVICE PARTITION FAILED";
            case  -19:  return  "KERNEL ARG INFO NOT AVAILABLE";
            case  -30:  return  "INVALID VALUE";
            case  -31:  return  "INVALID DEVICE TYPE";
            case  -32:  return  "INVALID PLATFORM";
            case  -33:  return  "INVALID DEVICE";
            case  -34:  return  "INVALID CONTEXT";
            case  -35:  return  "INVALID QUEUE PROPERTIES";
            case  -36:  return  "INVALID COMMAND QUEUE";
            case  -37:  return  "INVALID HOST PTR";
            case  -38:  return  "INVALID MEM OBJECT";
            case  -39:  return  "INVALID IMAGE FORMAT DESCRIPTOR";
            case  -40:  return  "INVALID IMAGE SIZE";
            case  -41:  return  "INVALID SAMPLER";
            case  -42:  return  "INVALID BINARY";
            case  -43:  return  "INVALID BUILD OPTIONS";
            case  -44:  return  "INVALID PROGRAM";
            case  -45:  return  "INVALID PROGRAM EXECUTABLE";
            case  -46:  return  "INVALID KERNEL NAME";
            case  -47:  return  "INVALID KERNEL DEFINITION";
            case  -48:  return  "INVALID KERNEL";
            case  -49:  return  "INVALID ARG INDEX";
            case  -50:  return  "INVALID ARG VALUE";
            case  -51:  return  "INVALID ARG SIZE";
            case  -52:  return  "INVALID KERNEL ARGS";
            case  -53:  return  "INVALID WORK DIMENSION";
            case  -54:  return  "INVALID WORK GROUP SIZE";
            case  -55:  return  "INVALID WORK ITEM SIZE";
            case  -56:  return  "INVALID GLOBAL OFFSET";
            case  -57:  return  "INVALID EVENT WAIT LIST";
            case  -58:  return  "INVALID EVENT";
            case  -59:  return  "INVALID OPERATION";
            case  -60:  return  "INVALID GL OBJECT";
            case  -61:  return  "INVALID BUFFER SIZE";
            case  -62:  return  "INVALID MIP LEVEL";
            case  -63:  return  "INVALID GLOBAL WORK SIZE";
            case  -64:  return  "INVALID PROPERTY";
            case  -65:  return  "INVALID IMAGE DESCRIPTOR";
            case  -66:  return  "INVALID COMPILER OPTIONS";
            case  -67:  return  "INVALID LINKER OPTIONS";
            case  -68:  return  "INVALID DEVICE PARTITION COUNT";
        }
        return "Unknown Error";
    }

    ComputeContext::ComputeContext()
        : device(getBestDevice()),
          context(device), 
          queue(context, device)
    { }

    DeviceCapabilities::DeviceCapabilities(cl::Device device)
        : memSize(device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()),
          maxAllocSize(device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>())
    {
        // TODO: find device capabilities
        std::cout << "Max Size: " << memSize << std::endl;
        std::cout << "Max Alloc Size: " << maxAllocSize << std::endl;
    }

    cl::Program buildProgram(cl::Context const& ctx, cl::Device dev, char const* filename)
    {
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
            // TODO: rethrow with build log as message
            std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev) << std::endl;
            exit(1);
        }

        return program;
    }

}
