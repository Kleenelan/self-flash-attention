EXE := ex_self_attn_cpu  ex_flash_attn_cpu

all: $(EXE)

LD_FLAGS :=  -L../local/lib64 -ls2fattn
INC := -I ../local/include

%: %.cpp
	g++ $< -o $@ $(INC) $(LD_FLAGS)

.PHONY: clean
clean:
	-rm -rf $(EXE)
