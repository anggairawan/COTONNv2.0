## Makefile for plain to dof abstraction controller converter

CC        = g++
CXXFLAGS 		= -Wall -Wextra -std=c++11 -O3 -DNDEBUG -DSCOTS_BDD
#
# scots 
#
SCOTSROOT		= ../ext/SCOTSv0.2
SCOTSINC		= -I$(SCOTSROOT)/src -I$(SCOTSROOT)/utils
#
# cudd 
#
CUDD		    =  -I/opt/local/include -L/opt/local/lib -lcudd

.PHONY: converter
	
TARGET = converter

all: $(TARGET)

converter: 
	$(CC) converter.cc -o converter $(CXXFLAGS) $(SCOTSINC) $(CUDD) 

clean:
	rm converter
