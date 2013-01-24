## DynamiCL

This is my attempt to write a fast HDR merge algorithm that runs on the GPU. It
is currently in development, in the stages of optimization.

![HDR example](https://github.com/KholdStare/dynamicl/raw/master/images/trafalgar-hdr.jpg)


## Features

* Based on the Mertens-Kautz-Van Reeth exposure fusion algorithm
  ([conference paper PDF](http://research.edm.uhasselt.be/%7Etmertens/papers/exposure_fusion_reduced.pdf)),
  like the [Enfuse](http://enblend.sourceforge.net/) tool.
* Uses OpenCL 1.2 to offload work to the GPU.
* Suitable for batch processing- separate concurrent threads for
  reading/merging/writing of files.

## TODOs

* Optimize DMA transfers to the GPU
  * Some specific commands allow concurrent DMA transfers and processing on the
    GPU (depends on vendor).
  * Memory has to be aligned and pages have to be pinned by the kernel to
    facilitate fastest transfer rates.
  * Devices have limits on the maximum size allocated per allocation in VRAM-
    need to transfer/process image in chunks for large images. This is the
    current bottleneck.
* Optimize OpenCL kernels.

