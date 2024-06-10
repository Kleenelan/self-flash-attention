EXE := self_attention flash_attention_cpu

all: $(EXE)

%: %.cpp
	g++ $< -o $@ -L ../local/lib64 -lsattn -I ../local/include

.PHONY: clean
clean:
	-rm -rf $(EXE)
