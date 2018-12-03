#
# compiler
#
#CC        = g++
CC       	  = clang++
# CXXFLAGS 		= -g -Wall -Wextra -std=c++11 -DSCOTS_BDD
CXXFLAGS 		= -g -Wall -Wextra -std=c++11 -O3 -DNDEBUG -DSCOTS_BDD
#
# scots 
#
SCOTSROOT		= ../../../ext/SCOTSv0.2
SCOTSINC		= -I$(SCOTSROOT)/src  -I$(SCOTSROOT)/utils/
#
# cudd 
#
CUDD		    =  -I/opt/local/include -L/opt/local/lib -lcudd

.PHONY: dcdc simulate bddconverter

TARGET = dcdc bddconverter simulate 

CONTROL = dcdc bddconverter

all: $(TARGET)

control: $(CONTROL)

dcdc: 
	$(CC) $(CXXFLAGS) $(SCOTSINC) $(CUDD) dcdc.cc -o dcdc

bddconverter: 
	$(CC) bddconverter.cc -o bddconverter $(CXXFLAGS) $(SCOTSINC) $(CUDD) 

simulate: 
	$(CC) $(CXXFLAGS) $(SCOTSINC) $(CUDD) simulate.cc -o simulate

clean:
	rm dcdc
