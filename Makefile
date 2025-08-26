make:
	g++ main.cpp -o build/RayTracer

run:
	"build/RayTracer.exe" > output/image.ppm

clean: 
	rmdir build
	mkdir build
	rmdir output
	mkdir output