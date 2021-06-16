# Subspace video stabilization

## Library Dependencies
- C++ 14
- OpenCV (with ffmpeg)
- Eigen
- Suite-Sparse
- Ceres Solver
- gflags
- glogger
- Python 3 (Assumed available)

## Primary Build Dependencies
- cmake
- make

## Install Make and CMake

`cmake` and `make` will be needed to both build the libraries required for the subSpace project and compile the subSpace project itself. 

First install `make` using,
`apt-get -y install make`

or,

` sudo apt-get install build-essential `

If there is a need for a specific version of cmake it can be obtained from the [downloads page](https://cmake.org/download/) of cmake and installed subsequently by following the [installation instructions](https://cmake.org/install/). Note that you will need to install `libssl-dev` using `sudo apt-get install libssl-dev`. If the build fails despite this, [this SO page](https://stackoverflow.com/questions/16248775/cmake-not-able-to-find-openssl-library) might help. However, if the version is not important, just run,

```  sudo apt-get -y install cmake ```

## Install Libraries

### Math Libraries

To my knowledge, C++ does not have a convenient package manager such as pip or anaconda are for python. Hence we will need to install all the libraries manually.

We will follow the instructions for installing `ceres` since the process for installing it takes care of all the dependencies other than `OpenCV + ffmpeg`.

Following the instructions on [this link](http://ceres-solver.org/installation.html) to install `Eigen`, `glags`, `glogger`, and `SuiteSparse`. Copied bellow as backup,

```bash
# google-glog + gflags
sudo apt-get install libgoogle-glog-dev libgflags-dev
# BLAS & LAPACK
sudo apt-get install libatlas-base-dev
# Eigen3
sudo apt-get install libeigen3-dev
# SuiteSparse and CXSparse (optional)
sudo apt-get install libsuitesparse-dev
# Download ceres source
git clone https://ceres-solver.googlesource.com/ceres-solver ceres
cd ceres
mkdir ceres-bin
cmake ../ceres-solver-2.0.0
make -j3
make test
# This step is marked optional in the instructions but will be needed in our case
sudo make install
```

### OpenCV + ffmpeg

A regular `opencv` installation along with a pre-existing `ffmpeg` installation had some left-over codec problems in my case. This was because even when opencv was available in the right locations for CMake to access, and the subSpace project was compiled correctly, on runnning the executable, there was an `ffmpeg` codec error when trying to read a video-file.

To resolve this, I had to recompile `ffmpeg` and opencv as explained in this [SO answer](https://stackoverflow.com/a/31130210/3642162). The instructions are below-

First Build `ffmpeg` from source, Download ffmpeg tarball [FFmpeg website](https://www.ffmpeg.org/download.html) and extract the directory in some location,

```bash
cd ffmpeg-folder
./configure --enable-pic --extra-ldexeflags=-pie
make -j5
sudo make install
```

Next install opencv from source. Download opencv source [from github](https://github.com/opencv/opencv) or from [source-forge](https://sourceforge.net/projects/opencvlibrary/files/). Extract the folder and `cd` into it. 

```bash
cd opencv-folder
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_QT=OFF -D WITH_V4L=ON -D CMAKE_SHARED_LINKER_FLAGS=-Wl,-Bsymbolic ..
make -j5 (under ffmpeg folder)
sudo make install
sudo sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
```

## Compile project

Download a copy of this project and extract the files or use `git clone`. Navigate to folder and run the below sequence of commands,

```bash
cd code
cmake CMakeLists.txt
make
```
Now if `make` was successful, there should be an executable called `SubspaceStab` in the project folder, and we are ready to use the project. If not, please create an issue here.

## Python requirments

There are no requirments beyond opencv and pre-installed libraries.

## Run Script usage 

After getting an executable should appear in the `build` directory we use a wrapper script to run the code. This is because the executable writes a collection of images to disk which we must seperately stich together into a video.

Script `subSpace.py` is a wrapper around the build for the exectuable generated after `make`. It takes a single input video path and an output video path. And an optional `--crop` flag.

```bash
    python3 subSpace.py -i "$video" -o "$out_name" --crop
```
or without cropping the black borders,

```bash
    python3 subSpace.py -i "$video" -o "$out_name"
```

Original script `run_all.py` runs on preselected datasets stiches together the final video purely using `ffmpeg`. `subSpace.py` stiches together the work using `opencv` video writer.

```bash
    python3 run_all.py
```

## References

- <a href="http://web.cecs.pdx.edu/~fliu/papers/tog2010.pdf">Subspace Video
Stabilization </a>
- <a href="http://gvv.mpi-inf.mpg.de/teaching/gvv_seminar_2012/papers/Content-Preserving%20Warps%20for%203D%20Video%20Stabilization.pdf">Content
Preserving Warps for 3D Video Stabilization</a>

## Related Work

- [List of Other](https://github.com/yaochih/awesome-video-stabilization) Video Stabilization Methods and Implementations

- L1 optimal camera paths video stabilization [implementation](https://github.com/ishank-juneja/L1-optimal-paths-Stabilization)

- Deep Online Video Stabilization [implementation](https://github.com/cxjyxxme/deep-online-video-stabilization-deploy)
