LIB_SATTN_SHARED := libsattn.so

SATTN_SRC_CPP := ./src/cpu_gemm.cpp ./src/utils.cpp ./src/cpu_softmax.cpp

SATTN_OBJ_CPP := $(SATTN_SRC_CPP:.cpp=.cpp.o)
SATTN_DEP_CPP := $(SATTN_SRC_CPP:.cpp=.cpp.d)

# SOLVER_OBJ_CU  := $(CU_SRC:.cu=.cu.o)
# SOLVER_DEP_CU  :=  $(CU_SRC:.cu=.cu.d)

CXX := g++

CXX_FLAGS := -fPIC
INC := -I./include

%.cpp.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INC) -c $^ -o $@ -MMD -MF $*.cpp.d -MP

$(LIB_SATTN_SHARED): $(SATTN_OBJ_CPP)
	$(CXX) -shared $^ -o $@


.PHONY: install
install:
	-mkdir -p ../local/lib64
	-mkdir -p ../local/include
	-cp libsattn.so ../local/lib64/
	-cp include/*.h ../local/include/

.PHONY: uninstall
uninstall:
	-rm -rf ../local/

.PHONY: clean
clean:
	-rm $(SATTN_OBJ_CPP) $(SATTN_DEP_CPP) $(LIB_SATTN_SHARED)
