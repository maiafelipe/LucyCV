CC=g++
#CFLAGS= -Wall 
CLIBS= -ggdb `pkg-config --cflags opencv` `pkg-config --libs opencv`
#
# Compiler flags:
#		-g	-- Enable debugging
#		-Wall	-- Turn on all warnings 
#		-D__USE_FIXED_PROTOTYPES__
#			-- Force the compiler to use the correct headers
#		-ansi	-- Don't use GNU extensions. Stick to ANSI C.

mlp: mlptest.cpp
	$(CC) $(CFLAGS) -o saida mlptest.cpp $(CLIBS)

rbf: rbftest.cpp
	$(CC) $(CFLAGS) -o saida rbftest.cpp $(CLIBS)

clean:
	rm -f saida
