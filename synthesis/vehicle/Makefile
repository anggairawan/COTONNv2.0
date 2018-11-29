#
# compiler
#
#CC        = g++
CC       	  = clang++
CXXFLAGS 		= -Wall -Wextra -std=c++11 -O3 -DNDEBUG
CXXDEBUG		= -g -Wall -Wextra -std=c++11 
#
# scots 
#
SCOTSROOT		= ../../ext/SCOTSv0.2
SCOTSINC		= -I$(SCOTSROOT)/src -I$(SCOTSROOT)/utils

.PHONY: vehicle simulate

TARGET = vehicle simulate

all: $(TARGET)

vehicle: 
	$(CC) $(CXXFLAGS) $(SCOTSINC) vehicle.cc -o vehicle

simulate: 
	$(CC) $(CXXFLAGS) $(SCOTSINC) simulate.cc -o simulate

clean:
	rm vehicle simulate
