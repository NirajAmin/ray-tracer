cpu:
	if not exist build mkdir build
	if not exist output mkdir output
	g++ src/main.cpp -o build/RayTracer.exe

run:
	"build/RayTracer.exe" > output/image.ppm

make gpu:
	if not exist build mkdir build
	if not exist output mkdir output
	nvcc src/main.cu src/cuda_render.cu src/main.cpp src/camera.cpp src/vec3.cpp src/ray.cpp src/color.cpp -o build/RayTracer_GPU.exe

run_gpu:
	"build/RayTracer_GPU.exe" > output/image.ppm

clean:
	if exist build rmdir /s /q build
	if exist output rmdir /s /q output
	mkdir build
	mkdir output
