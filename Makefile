make:
	g++ src/main.cpp -o build/RayTracer

run:
	"build/RayTracer.exe" > output/image.ppm

clean: 
	rmdir /s /q build
	mkdir build
	rmdir /s /q output
	mkdir output