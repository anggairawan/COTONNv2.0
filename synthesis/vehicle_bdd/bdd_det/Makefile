#
# compiler
#
#CC        = g++
CC       	  = clang++
CXXFLAGS 		= -Wall -Wextra -std=c++11 -O3 -DNDEBUG -DSCOTS_BDD
#
# scots 
#
SCOTSROOT		= ../../../ext/SCOTSv0.2
SCOTSINC		= -I$(SCOTSROOT)/src -I$(SCOTSROOT)/utils
#
# cudd 
#
CUDD		    =  -I/opt/local/include -L/opt/local/lib -lcudd

.PHONY: vehicle simulate bddconverter

TARGET = vehicle simulate

CONTROL = vehicle bddconverter

all: $(TARGET)

control: $(CONTROL)

vehicle: 
	$(CC) $(CXXFLAGS) $(SCOTSINC) $(CUDD) vehicle.cc -o vehicle

bddconverter: 
	$(CC) bddconverter.cc -o bddconverter $(CXXFLAGS) $(SCOTSINC) $(CUDD)

simulate: 
	$(CC) $(CXXFLAGS) $(SCOTSINC) $(CUDD) simulate.cc -o simulate

clean:
	rm vehicle *.scs *.bdd
