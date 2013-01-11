import os
import platform

env = Environment()  # Initialize the environment
env.Append(CPPFLAGS = [ '-O3', '-Wall', '-Werror', '-std=c++0x' ])
env.Append(LIBS = [ 'pthread',
            'boost_unit_test_framework',
            'OpenCL',
            'vigraimpex' ])

# debugging flags
debugflags = [ '-g', '-pg' ]
env.Append(CPPFLAGS =  debugflags)
env.Append(LINKFLAGS = debugflags)
	
bits = 'x86_64'
if platform.machine() != bits:
    bits = 'x86'

if os.environ['AMDAPPSDKROOT']:
    sdkroot = os.environ['AMDAPPSDKROOT']
    env.Append(CPPPATH = [ sdkroot + '/include' ])
    env.Append(LIBPATH = [ sdkroot + '/lib/' + bits ])

commonSource = ['utils.cpp',
                'cl_common.cpp',
                'cl_utils.cpp',
                'pending_image.cpp',
                'image_pyramid.cpp',
                'save_image.cpp' ]
mainSource = ['main.cpp', 'pyr_impl.cpp', 'merge_group.cpp' ]
testSource = ['test_suite.cpp']

mainSource.extend(commonSource)
testSource.extend(commonSource)

env.Program(target = 'dynamicl', source = mainSource)
env.Program(target = 'test_suite', source = testSource)
