NVCC        = nvcc

NVCC_FLAGS  = -std=c++11 -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	    = parallel_recommender
OBJ	    = main.o rec_gold.o rec_kernel.o ratings_util.o misc_util.o

default: $(EXE)

main.o: main.cpp recommendations_gold.h recommendations_kernel.h ratings_util.h misc_utls.h 
	$(NVCC) -c -o $@ main.cpp $(NVCC_FLAGS)

rec_gold.o: recommendations_gold.cpp recommendations_gold.h ratings_util.h
	$(NVCC) -c -o $@ recommendations_gold.cpp $(NVCC_FLAGS)

ratings_util.o: ratings_util.cpp ratings_util.h
	$(NVCC) -c -o $@ ratings_util.cpp $(NVCC_FLAGS)

rec_kernel.o: recommendations_kernel.cu recommendations_kernel.h ratings_util.h
	$(NVCC) -c -o $@ recommendations_kernel.cu $(NVCC_FLAGS)

misc_util.o: misc_utils.cpp misc_utls.h
	$(NVCC) -c -o $@ misc_utils.cpp $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
